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
import pickle

import network as simclr_net
from dataset import data_simclr
from gaussian_blur import GaussianBlur
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
		transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.RandomGrayscale(p = 0.2),
										GaussianBlur(kernel_size = 22),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	elif tr == 2:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomResizedCrop(size = 224,
																							  scale = (0.75, 1.0)),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.RandomGrayscale(p = 0.2),
										GaussianBlur(kernel_size = 22),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	elif tr == 3:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])

	elif tr == 4:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	elif tr == 5:
		transform = transforms.Compose([transforms.ToPILImage(),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	else:
		transform = transforms.Compose([transforms.ToPILImage(),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])

	return transform


def info_nce_loss(features, dev, temp):
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

	logits = logits / temp
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


def learn_unsupervised(args, simclrNet, device):
	numEpochs = args['epochs1']
	transform_train = get_train_transform(args["transform"])

	trainData = data_simclr(root = "./data", rng = args["rng"], train = True, nViews = 2, size = 224,
							transform = transform_train, fraction = 1, distort = args['distort'], adj = args['adj'],
							hyperTune = args["hypertune"])
	trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = args['batch_size'], shuffle = True,
												  num_workers = 2)

	optimizer = optimizers.SGD(simclrNet.backbone.parameters(), lr = args["lr"], weight_decay = args["weight_decay"],
							   momentum = 0.9)
	optimizer.add_param_group({'params': simclrNet.fc.parameters()})

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = int(1.25 * numEpochs),
														   eta_min = 0.1*args["lr"])
	show = False
	train_losses = []
	if args["resume"]:
		for ep in range(args["epochsRan"]):
			if ep > 8:
				scheduler.step()
	for ep in range(numEpochs):
		tqdmBar = tqdm.tqdm(trainDataLoader)
		b = 0
		avg_loss = 0.0
		for _, images, _ in tqdmBar:
			b += 1
			optimizer.zero_grad()
			images = torch.cat(images, dim = 0)
			if show:
				unorm = UnNormalize(mean = mean, std = std)
				im1 = transforms.ToPILImage()(unorm(images[0]))
				im1.show()
				im2 = transforms.ToPILImage()(unorm(images[args['batch_size']]))
				im2.show()
				show = False
			images = images.to(device)
			features = simclrNet(images)
			logits, labels = info_nce_loss(features = features, dev = device, temp = args["temperature"])
			loss = nn.CrossEntropyLoss().to(device)(logits, labels)
			avg_loss = (avg_loss * (b - 1) + loss.item()) / b
			loss.backward()
			optimizer.step()
			tqdmBar.set_description("Epoch: {:d}/{:d}, Loss: {:.6f}, LR: {:.8f}".format(ep + 1, numEpochs, avg_loss,
																						optimizer.param_groups[0][
																							'lr']))

		train_losses.append(avg_loss)
		if args["resume"] or ep > 8:
			scheduler.step()
		if args["saveRate"] != -1 and (ep + 1) % args["saveRate"] == 0 and args["save"]:
			fileName = args["saveName"] + "_unsupervised_" + str(ep + 1) + ".pt"
			torch.save(simclrNet.state_dict(), fileName, _use_new_zipfile_serialization = False)
	if args["save"]:
		fileName = args["saveName"] + "_unsupervised_final.pt"
		torch.save(simclrNet.state_dict(), fileName, _use_new_zipfile_serialization = False)
		fileName = args["saveName"] + "_train_losses.pickle"
		f = open(fileName, "wb")
		pickle.dump(train_losses, f, protocol = pickle.DEFAULT_PROTOCOL)
		f.close()


def learn_supervised(args, simclrNet, device, k):
	transform_train = get_train_transform(args["transform"])

	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
										 transforms.Normalize(mean, std)])
	trainSet = data_simclr(root = "./data", train = True, transform = transform_train, split = "super", size = 224,
						   fraction = args["frac"], hyperTune = args["hypertune"], rng = args["rng"])
	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = args['batch_size'], shuffle = True)

	testSet = data_simclr(root = "./data", train = False, transform = transform_test, split = "super", size = 224,
						  hyperTune = args["hypertune"], rng = args["rng"])
	testLoader = torch.utils.data.DataLoader(testSet, batch_size = args['batch_size'], shuffle = False)
	if args["freeze_backbone"]:
		simclrNet.freeze_feat()
	else:
		simclrNet.freeze_head()
	pytorch_total_params = sum(p.numel() for p in simclrNet.parameters())
	pytorch_total_params_train = sum(p.numel() for p in simclrNet.parameters() if p.requires_grad)
	print(pytorch_total_params, pytorch_total_params_train)
	net = simclrNet.to(device)

	optimizer = torch.optim.SGD(net.classifier_fc.parameters(), lr = args["lr_ft"], weight_decay = args["weight_decay"])
	if not args["freeze_backbone"]:
		optimizer.add_param_group({'params': simclrNet.backbone.parameters()})

	numEpochsS = args['epochs2']

	for ep in range(numEpochsS):
		ep_id = 0
		tot_loss = 0
		tqdmBar = tqdm.tqdm(trainLoader)
		for _, images, labels in tqdmBar:
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			logits = net.classify(images)
			loss = nn.CrossEntropyLoss()(logits, labels)
			tot_loss += loss.item()
			ep_id += 1
			tqdmBar.set_description("Repetition: {:d}/{:d} Epoch: {:d}/{:d} Loss: {:.4f}".format(k, args["supervisedRep"],
																				ep + 1, numEpochsS, tot_loss / ep_id))
			loss.backward()
			optimizer.step()
		if ep % 5 == 0:
			optimizer.param_groups[0]['lr'] *= 0.7

	net.eval()

	if args["save"]:
		fileName = args["saveName"] + "_test_predictions.csv"
		csvFileTest = open(fileName, "w")
		csvWriterTest = csv.writer(csvFileTest)
		csvWriterTest.writerow(["Index", "True Label", "Predicted Label"])

		fileName = args["saveName"] + "_train_predictions.csv"
		csvFileTrain = open(fileName, "w")
		csvWriterTrain = csv.writer(csvFileTrain)
		csvWriterTrain.writerow(["Index", "True Label", "Predicted Label"])

		fileName = args["saveName"] + "_train_indices.pickle"
		selectedIndicesFile = open(fileName, "wb")
		pickle.dump(trainSet.indicesSelected, selectedIndicesFile, pickle.DEFAULT_PROTOCOL)
		selectedIndicesFile.close()

	top1acc = 0
	top5acc = 0
	totTrainPoints = 0
	for _, (indices, images, labels) in enumerate(trainLoader):
		images = images.to(device)
		labels = labels.to(device)
		with torch.no_grad():
			logits = net.classify(images)
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
			logits = net.classify(images)
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
		torch.save(simclrNet.state_dict(), fileName, _use_new_zipfile_serialization = False)
		csvFileTrain.close()
		csvFileTest.close()

	return top1acc, top5acc


def set_seed(sd):
	if sd == -1:
		sd = np.random.randint(0, 65536)
	torch.manual_seed(sd)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	rng = np.random.default_rng(sd)
	return rng


def train_unsupervised_and_supervised(args):
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
	network = simclr_net.SimClRNet(numClasses = 12).to(device)
	if args["resume"]:
		if args["resumeFile"] == "":
			raise RuntimeError("No file provided for model to start from.")
		if args["epochsRan"] == -1:
			raise RuntimeError("Specify number of epochs ran for model which should be trained further.")
		network.load_state_dict(torch.load(outputDirectory + args["resumeFile"]))
		args["saveName"] = outputDirectory + args["resumeFile"]
	network.freeze_classifier()
	if args["save"]:
		configFileName = args["saveName"] + "_config.pickle"
		configFile = open(configFileName, "wb")
		pickle.dump(args, configFile, pickle.DEFAULT_PROTOCOL)
		# print(args, file = configFile)
		configFile.close()

	learn_unsupervised(args = args, simclrNet = network, device = device)
	pytorch_total_params = sum(p.numel() for p in network.parameters())
	pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
	print(pytorch_total_params, pytorch_total_params_train)
	learn_supervised(args = args, simclrNet = network, device = device, k = 1)


def evaluate_trained_network(args):
	print(torch.cuda.get_device_name(0))
	device = torch.device('cuda:0')
	args["start"] = datetime.datetime.now()
	args["saveName"] = outputDirectory + args["saveName"]
	if not args["resume"]:
		raise RuntimeError("Set resume flag and specify running file")
	if args["supervisedRep"] < 1:
		raise RuntimeError("Number of repetitions for supervised training must be > 0. Use \'-rep\' option to set.")
	if args["resume"]:
		if args["resumeFile"] == "":
			raise RuntimeError("No file provided for model to start from.")
		if args["saveName"] == "":
			args["saveName"] = outputDirectory + args["resumeFile"]
	saveName = args["saveName"]
	print("Starting from ", args["resumeFile"])
	accuracies = []
	for i in range(args["supervisedRep"]):
		args["saveName"] = saveName + "_" + str(i + 1)
		rng = set_seed(args["seed"])
		args["rng"] = rng
		network = simclr_net.SimClRNet(numClasses = 12).to(device)
		network.load_state_dict(torch.load(outputDirectory + args["resumeFile"]))
		network.freeze_classifier()
		if args["save"]:
			configFileName = args["saveName"] + "_config.pickle"
			configFile = open(configFileName, "wb")
			pickle.dump(args, configFile, pickle.DEFAULT_PROTOCOL)
			# print(args, file = configFile)
			configFile.close()
		pytorch_total_params = sum(p.numel() for p in network.parameters())
		pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
		print(pytorch_total_params, pytorch_total_params_train)
		top1, top5 = learn_supervised(args = args, simclrNet = network, device = device, k = i + 1)
		accuracies.append(top1)
	print(accuracies, np.mean(np.asarray(accuracies)), np.std(np.asarray(accuracies)))
	fileName = saveName + "_test_accuracies.csv"
	acc_file = open(fileName, "w")
	csv_acc = csv.writer(acc_file)
	for acc in accuracies:
		csv_acc.writerow([acc])
	acc_file.close()


if __name__ == "__main__":
	if not os.path.isdir(outputDirectory):
		os.mkdir(outputDirectory)
	simclr_args = vars(parser.get_parser("SimCLR Parser"))
	saveName = simclr_args["saveName"]
	num_reps = simclr_args["supervisedRep"]
	simclr_args["supervisedRep"] = 1
	train_unsupervised_and_supervised(args = simclr_args)

	simclr_args["resume"] = True
	simclr_args["resumeFile"] = saveName + "_unsupervised_final.pt"
	simclr_args["saveName"] = saveName + "_eval"
	simclr_args["supervisedRep"] = num_reps
	evaluate_trained_network(args = simclr_args)
