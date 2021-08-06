import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optimizers
import tqdm
import numpy as np
import os
import csv
import datetime

import network as net
from dataset_toybox import data_toybox
from dataset_cifar import data_cifar10
import parser

outputDirectory = "./output/"
mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)

class UnNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""
		for t, m, s in zip(tensor, self.mean, self.std):
			t.mul_(s).add_(m)
			# The normalize code -> t.sub_(m).div_(s)
		return tensor


def get_train_transform(tr):
	s = 1
	color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
	if tr == 1:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
										transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.RandomGrayscale(p = 0.2),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	elif tr == 2:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
										transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])

	elif tr == 3:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
										transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	elif tr == 4:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
							transforms.RandomHorizontalFlip(p = 0.5),
							transforms.ToTensor(),
							transforms.Normalize(mean, std)])
	else:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
							transforms.RandomHorizontalFlip(p = 0.5),
							transforms.RandomApply([color_jitter], p = 0.8),
							transforms.ToTensor(),
							transforms.Normalize(mean, std)])

	return transform


def loss_fn(x, y):
	x = F.normalize(x, dim=-1, p=2)
	y = F.normalize(y, dim=-1, p=2)
	loss = 2 - 2 * (x * y).sum(dim=-1)
	return loss


def info_nce_loss(features, dev):
	batchSize = features.shape[0] / 2
	labels = torch.cat([torch.arange(batchSize) for _ in range(2)], dim = 0)
	labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
	labels = labels.to(dev)

	features = F.normalize(features, dim = 1)

	similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1))
	# discard the main diagonal from both: labels and similarities matrix
	mask = torch.eye(labels.shape[0], dtype = torch.bool).to(dev)
	labels = labels[~mask].view(labels.shape[0], -1)
	similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
	positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
	negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

	logits = torch.cat([positives, negatives], dim = 1)
	labels = torch.zeros(logits.shape[0], dtype = torch.long).to(dev)

	logits = logits / 0.5
	return logits, labels


def calc_accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batchSize = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		target_reshaped = torch.reshape(target, (1, -1)).repeat(maxk, 1)
		correct_top_k = torch.eq(pred, target_reshaped)
		pred_1 = pred[0]
		res = []
		for k in topk:
			correct_k = correct_top_k[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(torch.mul(correct_k, 100.0 / batchSize))
		return res, pred_1


def get_dist_mat(act_a, act_b):
	aR = torch.repeat_interleave(act_a, act_b.shape[0], dim = 0)
	bR = act_b.repeat(act_a.shape[0], 1)
	dist_mat = torch.sqrt(torch.pow(aR - bR, 2).sum(dim = 1))
	dist_mat = dist_mat.view(act_a.shape[0], -1)
	return dist_mat


def get_dist(act_a, act_b):
	distMat = None
	for i in range(act_a.shape[0]):
		dists = torch.sqrt(torch.pow(act_a[i] - act_b, 2).sum(dim = 1)).unsqueeze(0)
		if distMat is None:
			distMat = dists
		else:
			distMat = torch.cat((distMat, dists), dim = 0)

	return distMat


def knn_eval(network, trainData, testData):
	trainActs = None
	trainLabels = None
	for _, (trainIms, _), labels in trainData:
		with torch.no_grad():
			activations = network.encoder_backbone(trainIms.cuda())
			if trainActs is None:
				trainActs = activations
				trainLabels = labels
			else:
				trainActs = torch.cat((trainActs, activations))
				trainLabels = torch.cat((trainLabels, labels))
	# print("Train activation size:", trainActs.shape, trainLabels.shape)
	testActs = None
	testLabels = None
	i = 0
	for _, (_, testIms, labels) in enumerate(testData):
		with torch.no_grad():
			activations = network.encoder_backbone(testIms.cuda())
			i += 1
			if testActs is None:
				testActs = activations
				testLabels = labels
			else:
				testActs = torch.cat((testActs, activations))
				testLabels = torch.cat((testLabels, labels))
	# print("Test activation size:", testActs.shape, testLabels.shape)

	dist_matrix = get_dist(act_a = testActs.cuda(), act_b = trainActs.cuda())
	# print("Distance matrix:", dist_matrix.shape)
	topkDist, topkInd = torch.topk(dist_matrix, k = 200, dim = 1, largest = False)
	# print(topkInd.shape)

	# print(topkInd.squeeze().cpu())
	preds = trainLabels[topkInd.squeeze()]
	# print(preds.shape)
	predsMode, _ = torch.mode(preds, dim = 1)
	# print(predsMode.shape, predsMode[:50])
	# print(torch.eq(preds, testLabels).float().sum(),
	acc = 100 * torch.eq(predsMode, testLabels).float().sum()/testLabels.shape[0]
	return acc.numpy()


def learn_unsupervised(args, network, device):
	numEpochs = args['epochs1']
	transform_train = get_train_transform(args["transform"])
	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(),
										 transforms.Normalize(mean, std)])

	if args['dataset'] == "toybox":
		trainData = data_toybox(root = "./data", rng = args["rng"], train = True, nViews = 2, size = 224,
							transform = transform_train, fraction = args['frac1'], distort = args['distort'], adj = args['adj'],
							hyperTune = args["hypertune"])
		testSet = data_toybox(root = "./data", train = False, transform = transform_test, split = "super", size = 224,
							  hyperTune = args["hypertune"], rng = args["rng"])
	else:
		trainData = data_cifar10(root = "./data", rng = args["rng"], train = True, nViews = 2, size = 224,
							transform = transform_train, fraction = args['frac1'], distort = args['distort'], adj = args['adj'],
							hyperTune = args["hypertune"])
		testSet = data_cifar10(root = "./data", train = False, transform = transform_test, split = "super", size = 224,
							   hyperTune = args["hypertune"], rng = args["rng"])

	trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = args['batch_size'], shuffle = True,
												  num_workers = 2)
	testLoader = torch.utils.data.DataLoader(testSet, batch_size = args['batch_size'], shuffle = False)

	optimizer = optimizers.SGD(network.encoder_backbone.parameters(), lr = args["lr"], weight_decay = 0.0005,
							   momentum = 0.9)
	optimizer.add_param_group({'params': network.encoder_projection.parameters()})
	optimizer.add_param_group({'params': network.encoder_prediction.parameters()})

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = numEpochs - 10, eta_min = 0.01)
	show = False
	for ep in range(numEpochs):
		tqdmBar = tqdm.tqdm(trainDataLoader)
		b = 0
		avg_loss = 0.0
		for _, (images1, images2), _ in tqdmBar:
			b += 1
			optimizer.zero_grad()
			if show:
				unorm = UnNormalize(mean = mean, std = std)
				im1 = transforms.ToPILImage()(unorm(images1[0]))
				im1.show()
				im2 = transforms.ToPILImage()(unorm(images2[0]))
				im2.show()
				show = False
			images1 = images1.to(device)
			images2 = images2.to(device)
			# print(images1.shape, images2.shape)
			features1 = network.encoder_forward(images1)
			features2 = network.encoder_forward(images2)

			targets1 = network.target_forward(images2)
			targets2 = network.target_forward(images1)

			loss = loss_fn(features1, targets1.detach())
			loss += loss_fn(features2, targets2.detach())
			loss = loss.mean()
			avg_loss = (avg_loss * (b - 1) + loss) / b
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				network.update_target_network()
			tqdmBar.set_description("Epoch: {:d}/{:d}, Loss: {:.6f}, LR: {:.8f}, b: {:.4f}".format(ep + 1, numEpochs, avg_loss,
																						optimizer.param_groups[0][
																							'lr'], network.beta))
		network.update_momentum(ep + 1, numEpochs)
		if ep % 50 == 0:
			knn_acc = knn_eval(network = network, trainData = trainDataLoader, testData = testLoader)
			print("knn accuracy:", knn_acc)
		if ep > 8:
			scheduler.step()
		if args["saveRate"] != -1 and (ep + 1) % args["saveRate"] == 0 and args["save"]:
			fileName = args["saveName"] + "_unsupervised_" + str(ep + 1) + ".pt"
			torch.save(network.state_dict(), fileName, _use_new_zipfile_serialization = False)
	if args["save"]:
		fileName = args["saveName"] + "_unsupervised_final.pt"
		torch.save(network.state_dict(), fileName, _use_new_zipfile_serialization = False)


def learn_supervised(args, network, device):
	transform_train = get_train_transform(args["transform"])

	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(),
										 transforms.Normalize(mean, std)])

	if args['dataset'] == "toybox":
		trainSet = data_toybox(root = "./data", train = True, transform = transform_train, split = "super", size = 224,
						   fraction = args["frac2"], hyperTune = args["hypertune"], rng = args["rng"])

		testSet = data_toybox(root = "./data", train = False, transform = transform_test, split = "super", size = 224,
							  hyperTune = args["hypertune"], rng = args["rng"])

	else:
		trainSet = data_cifar10(root = "./data", train = True, transform = transform_train, split = "super", size = 224,
							   fraction = args["frac2"], hyperTune = args["hypertune"], rng = args["rng"])

		testSet = data_cifar10(root = "./data", train = False, transform = transform_test, split = "super", size = 224,
							  hyperTune = args["hypertune"], rng = args["rng"])

	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = args['batch_size'], shuffle = True)

	testLoader = torch.utils.data.DataLoader(testSet, batch_size = args['batch_size'], shuffle = False)
	pytorch_total_params = sum(p.numel() for p in network.parameters())
	pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
	print(pytorch_total_params, pytorch_total_params_train)
	network = network.to(device)

	optimizer = torch.optim.SGD(network.classifier_fc.parameters(), lr = args["lr_ft"], weight_decay = 0.00005)
	if not args["freeze_backbone"]:
		optimizer.add_param_group({'params': network.backbone.parameters()})

	numEpochsS = args['epochs2']

	for ep in range(numEpochsS):
		ep_id = 0
		tot_loss = 0
		tqdmBar = tqdm.tqdm(trainLoader)
		for _, images, labels in tqdmBar:
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			logits = network.classify(images)
			loss = nn.CrossEntropyLoss()(logits, labels)
			tot_loss += loss.item()
			ep_id += 1
			tqdmBar.set_description("Epoch: {:d}/{:d} Loss: {:.4f} LR: {:.8f}".format(ep + 1, numEpochsS, tot_loss / ep_id,
																					  optimizer.param_groups[0]['lr']))
			loss.backward()
			optimizer.step()
		if ep % 20 == 19 and ep > 0:
			optimizer.param_groups[0]['lr'] *= 0.7

	network.eval()

	if args["save"]:
		fileName = args["saveName"] + "_test_predictions.csv"
		csvFileTest = open(fileName, "w")
		csvWriterTest = csv.writer(csvFileTest)
		csvWriterTest.writerow(["Index", "True Label", "Predicted Label"])

		fileName = args["saveName"] + "_train_predictions.csv"
		csvFileTrain = open(fileName, "w")
		csvWriterTrain = csv.writer(csvFileTrain)
		csvWriterTrain.writerow(["Index", "True Label", "Predicted Label"])

	top1acc = 0
	top5acc = 0
	totTrainPoints = 0
	for _, (indices, images, labels) in enumerate(trainLoader):
		images = images.to(device)
		labels = labels.to(device)
		with torch.no_grad():
			logits = network.classify(images)
		top, pred = calc_accuracy(logits, labels, topk = (1, 5))
		top1acc += top[0].item() * pred.shape[0]
		top5acc += top[1].item() * pred.shape[0]
		totTrainPoints += pred.shape[0]
		if args["save"]:
			pred, labels, indices = pred.cpu().numpy(), labels.cpu().numpy(), indices.cpu().numpy()
			for idx in range(pred.shape[0]):
				row = [indices[idx], labels[idx], pred[idx]]
				csvWriterTrain.writerow(row)
	top1acc /= totTrainPoints
	top5acc /= totTrainPoints

	print("Train Accuracies 1 and 5:", top1acc, top5acc)

	top1acc = 0
	top5acc = 0
	totTestPoints = 0
	for _, (indices, images, labels) in enumerate(testLoader):
		images = images.to(device)
		labels = labels.to(device)
		with torch.no_grad():
			logits = network.classify(images)
		top, pred = calc_accuracy(logits, labels, topk = (1, 5))
		top1acc += top[0].item() * indices.size()[0]
		top5acc += top[1].item() * indices.size()[0]
		totTestPoints += indices.size()[0]
		if args["save"]:
			pred, labels, indices = pred.cpu().numpy(), labels.cpu().numpy(), indices.cpu().numpy()
			for idx in range(pred.shape[0]):
				row = [indices[idx], labels[idx], pred[idx]]
				csvWriterTest.writerow(row)
	top1acc /= totTestPoints
	top5acc /= totTestPoints

	print("Test Accuracies 1 and 5:", top1acc, top5acc)

	if args["save"]:
		fileName = args["saveName"] + "_supervised.pt"
		torch.save(network.state_dict(), fileName, _use_new_zipfile_serialization = False)
		csvFileTrain.close()
		csvFileTest.close()
	return top1acc


def set_seed(sd):
	if sd == -1:
		sd = np.random.randint(0, 65536)
	print("Setting seed to", sd)
	torch.manual_seed(sd)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	rng = np.random.default_rng(sd)
	return rng


def run_experiments(args):
	print(torch.cuda.get_device_name(0))
	args["start"] = datetime.datetime.now()
	rng = set_seed(args["seed"])
	args["rng"] = rng
	if args["saveName"] == "":
		if args["distort"] == "transform":
			args["saveName"] = "trained_model_cropped_" + args["distort"] + "_" + str(args["adj"])
		else:
			args["saveName"] = "trained_model_cropped_" + args["distort"]
	args["saveName"] = outputDirectory + args["saveName"]
	device = torch.device('cuda:0')
	network = net.BYOLNet(numClasses = 12).to(device)
	if args["resume"]:
		if args["resumeFile"] == "":
			raise RuntimeError("No file provided for model to start from.")
		network.load_state_dict(torch.load(outputDirectory + args["resumeFile"]))
		args["saveName"] = outputDirectory + args["resumeFile"]
	network.freeze_classifier()
	if args["save"]:
		configFileName = args["saveName"] + "_config.txt"
		configFile = open(configFileName, "w")
		print(args, file = configFile)
		configFile.close()

	network.freeze_classifier()
	network.freeze_target_backbone()
	network.freeze_target_projection()
	network.unfreeze_encoder_backbone()
	network.unfreeze_encoder_projection()
	network.unfreeze_encoder_prediction()
	network.train()
	learn_unsupervised(args = args, network = network, device = device)
	pytorch_total_params = sum(p.numel() for p in network.parameters())
	pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
	print(pytorch_total_params, pytorch_total_params_train)
	network.unfreeze_classifier()
	network.freeze_encoder_backbone()
	network.freeze_encoder_projection()
	network.freeze_encoder_prediction()
	network.freeze_target_backbone()
	network.freeze_target_projection()
	acc = learn_supervised(args = args, network = network, device = device)
	return acc

def eval_network(args):
	assert args["resume"]
	print(torch.cuda.get_device_name(0))
	args["start"] = datetime.datetime.now()
	rng = set_seed(args["seed"])
	args["rng"] = rng
	if args["saveName"] == "":
		if args["distort"] == "transform":
			args["saveName"] = "trained_model_cropped_" + args["distort"] + "_" + str(args["adj"])
		else:
			args["saveName"] = "trained_model_cropped_" + args["distort"]
	args["saveName"] = outputDirectory + args["saveName"]
	device = torch.device('cuda:0')
	network = net.BYOLNet(numClasses = 12).to(device)
	if args["resume"]:
		if args["resumeFile"] == "":
			raise RuntimeError("No file provided for model to start from.")
		network.load_state_dict(torch.load(outputDirectory + args["resumeFile"]))
		print("Loading network from", args["resumeFile"])
		args["saveName"] = outputDirectory + args["resumeFile"]
	pytorch_total_params = sum(p.numel() for p in network.parameters())
	pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
	print(pytorch_total_params, pytorch_total_params_train)
	network.unfreeze_classifier()
	network.freeze_encoder_backbone()
	network.freeze_encoder_projection()
	network.freeze_encoder_prediction()
	network.freeze_target_backbone()
	network.freeze_target_projection()

	pytorch_total_params = sum(p.numel() for p in network.parameters())
	pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
	print(pytorch_total_params, pytorch_total_params_train)

	acc = learn_supervised(args = args, network = network, device = device)
	return acc


if __name__ == "__main__":
	if not os.path.isdir(outputDirectory):
		os.mkdir(outputDirectory)
	byol_args = vars(parser.get_parser("SimCLR Parser"))
	accs = []
	fileName = ""
	if byol_args['supervisedRep'] > 0:
		assert byol_args['save'] or os.path.isfile(outputDirectory + byol_args['resumeFile'])
		if not os.path.isfile(outputDirectory + byol_args['resumeFile']):
			fileName = byol_args["saveName"] + "_unsupervised_final.pt"
		else:
			fileName = byol_args['resumeFile']
	sup_acc = run_experiments(args = byol_args)
	accs.append(sup_acc)
	if byol_args['supervisedRep'] > 0:
		byol_args['resume'] = True
		byol_args['resumeFile'] = fileName
		byol_args['save'] = False
		for rep in range(byol_args["supervisedRep"]):
			print("--------------------------------------------------------------------------")
			print("Run ", str(rep + 1), "of ", str(byol_args["supervisedRep"]))
			byol_args['seed'] = -1
			sup_acc = eval_network(args = byol_args)
			accs.append(sup_acc)
	print("Accuracies:", accs)
	print("Mean:", np.mean(np.asarray(accs)), "Std:", np.std(np.asarray(accs)))


