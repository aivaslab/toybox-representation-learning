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
from dataset import data_simclr
import parser

outputDirectory = "./output/"


def get_train_transform(tr):
	s = 1
	color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
	if tr == 1:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size = 224, padding = 25),
										  transforms.RandomHorizontalFlip(p = 0.5),
										  transforms.RandomApply([color_jitter], p = 0.8),
										  transforms.RandomGrayscale(p = 0.2),
										  transforms.ToTensor(),
										  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	elif tr == 2:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size = 224, padding = 25),
										  transforms.RandomHorizontalFlip(p = 0.5),
										  transforms.RandomApply([color_jitter], p = 0.8),
										  transforms.ToTensor(),
										  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	elif tr == 3:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size = 224, padding = 25),
							transforms.RandomHorizontalFlip(p = 0.5),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	elif tr == 4:
		transform = transforms.Compose([transforms.ToPILImage(),
							transforms.RandomHorizontalFlip(p = 0.5),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	else:
		transform = transforms.Compose([transforms.ToPILImage(),
							transforms.RandomHorizontalFlip(p = 0.5),
							transforms.RandomApply([color_jitter], p = 0.8),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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


def learn_unsupervised(args, network, device):
	numEpochs = args['epochs1']
	transform_train = get_train_transform(args["transform"])

	trainData = data_simclr(root = "./data", rng = args["rng"], train = True, nViews = 2, size = 224,
							transform = transform_train, fraction = 0.1, distort = args['distort'], adj = args['adj'],
							hyperTune = args["hypertune"])
	trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = args['batch_size'], shuffle = True,
												  num_workers = 4)

	optimizer = optimizers.SGD(network.encoder_backbone.parameters(), lr = args["lr"], weight_decay = 0.0005,
							   momentum = 0.9)
	optimizer.add_param_group({'params': network.encoder_projection.parameters()})
	optimizer.add_param_group({'params': network.encoder_prediction.parameters()})

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 250, eta_min = 0.001)
	show = False
	for ep in range(numEpochs):
		tqdmBar = tqdm.tqdm(trainDataLoader)
		b = 0
		avg_loss = 0.0
		for _, (images1, images2), _ in tqdmBar:
			b += 1
			optimizer.zero_grad()
			if show:
				im1 = transforms.ToPILImage()(images1[0])
				im1.show()
				im2 = transforms.ToPILImage()(images1[args['batch_size']])
				im2.show()
				show = False
			images1 = images1.to(device)
			images2 = images2.to(device)
			# print(images1.shape, images2.shape)
			features1 = network.encoder_forward(images1)
			features2 = network.encoder_forward(images2)
			with torch.no_grad():
				targets1 = network.target_forward(images2)
				targets2 = network.target_forward(images1)

			loss = loss_fn(features1, targets1)
			loss += loss_fn(features2, targets2)
			loss = loss.mean()
			avg_loss = (avg_loss * (b - 1) + loss) / b
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				network.update_target_network()
			tqdmBar.set_description("Epoch: {:d}/{:d}, Loss: {:.6f}, LR: {:.8f}".format(ep + 1, numEpochs, avg_loss,
																						optimizer.param_groups[0][
																							'lr']))
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

	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
										 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainSet = data_simclr(root = "./data", train = True, transform = transform_train, split = "super", size = 224,
						   fraction = args["frac"], hyperTune = args["hypertune"], rng = args["rng"])
	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = args['batch_size'], shuffle = True)

	testSet = data_simclr(root = "./data", train = False, transform = transform_test, split = "super", size = 224,
						  hyperTune = args["hypertune"], rng = args["rng"])
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
			tqdmBar.set_description("Epoch: {:d}/{:d} Loss: {:.4f}".format(ep + 1, numEpochsS, tot_loss / ep_id))
			loss.backward()
			optimizer.step()
		if ep % 5 == 0:
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


def set_seed(sd):
	torch.manual_seed(sd)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	rng = np.random.default_rng(sd)
	return rng


def run_experiments(args):
	print(torch.cuda.get_device_name(0))
	args["start"] = datetime.datetime.now()
	rng = set_seed(0)
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
	learn_supervised(args = args, network = network, device = device)


if __name__ == "__main__":
	if not os.path.isdir(outputDirectory):
		os.mkdir(outputDirectory)
	simclr_args = vars(parser.get_parser("SimCLR Parser"))
	run_experiments(args = simclr_args)
