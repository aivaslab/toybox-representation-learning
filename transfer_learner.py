import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import torch
import time
import csv
import pickle
import math
import network as simclr_net
import argparse
import os
import torch.nn as nn
import utils
import tqdm
import comb_network
import cifar10_frac

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)


def get_parser(desc):
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument('--model', '-m', required = True, type = str)
	parser.add_argument('--lr', '-lr', required = True, type = float)
	parser.add_argument('--epochs', '-e', required = True, type = int)
	parser.add_argument('--fraction', '-f', default = 0.1, type = float)
	parser.add_argument('--dataset', '-data', default = "cifar10", type = str)
	parser.add_argument('--combined', '-c', default = False, action = "store_true")
	parser.add_argument('--batch-size', '-b', default = 128, type = int)

	return parser.parse_args()


if __name__ == "__main__":

	# network = simclr_net.SimClRNet(numClasses = 12).cuda()
	args = vars(get_parser("Face Learner"))
	if args['combined'] is True:
		network = comb_network.SimClRNet(numClasses = 12).cuda()
	else:
		network = simclr_net.SimClRNet(numClasses = 12).cuda()

	fileName = "./pose_models/" + args['model']
	assert(os.path.isfile(fileName))
	network.load_state_dict(torch.load(fileName))
	network.freeze_all_params()
	network.eval()
	featSize = network.classifier_fc.in_features
	if args['dataset'] == "cifar10":
		face_classifier = nn.Sequential(nn.Linear(featSize, featSize//2), nn.ReLU(), nn.Linear(featSize//2, 10)).cuda()
	else:
		face_classifier = nn.Sequential(nn.Linear(featSize, featSize // 2), nn.ReLU(),
										nn.Linear(featSize // 2, 100)).cuda()

	trainTransform = transforms.Compose([transforms.ToPILImage(),
										 transforms.Resize(224),
										 transforms.RandomHorizontalFlip(p = 0.5),
										 transforms.RandomCrop(size = 224, padding = 5),
										 transforms.ToTensor(),
										 transforms.Normalize(mean, std)])

	if args['dataset'] == "cifar10":
		trainData = cifar10_frac.fCIFAR10(root = "./data", train = True, transform = trainTransform, download = True, frac =
									args['fraction'])
	else:
		trainData = cifar10_frac.fCIFAR100(root = "./data", train = True, transform = trainTransform, download = True, frac =
									args['fraction'])

	trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = args['batch_size'], shuffle = True,
													  num_workers = 4)
	if args['dataset'] == "cifar10":
		face_data_test = cifar10_frac.fCIFAR10(root = "./data", train = False, download = True,
									  transform = trainTransform)
	else:
		face_data_test = cifar10_frac.fCIFAR100(root = "./data", train = False, download = True,
											   transform = trainTransform)

	testDataLoader = torch.utils.data.DataLoader(face_data_test, batch_size = args['batch_size'], shuffle = True,
													  num_workers = 4)
	optimizer = torch.optim.SGD(face_classifier.parameters(), lr = args['lr'], weight_decay = 1e-6, momentum = 0.9)
	# optimizer.add_param_group({'params': network.backbone.parameters()})

	numEpochs = args['epochs']
	for ep in range(numEpochs):
		avg_loss = 0
		b = 0
		tqdmBar = tqdm.tqdm(trainDataLoader)
		for idx, (images, targets) in enumerate(tqdmBar):
			optimizer.zero_grad()
			b += 1
			images = images.cuda()
			targets = targets.to(torch.device('cuda:0'))
			feats = network.backbone(images)
			logits = face_classifier(feats)
			loss = nn.CrossEntropyLoss()(logits, targets)
			avg_loss = (avg_loss * (b - 1) + loss.item()) / b
			loss.backward()
			optimizer.step()
			tqdmBar.set_description("Epoch: {:d}/{:d} Loss: {:.4f}, LR: {:.8f}".format(ep + 1, numEpochs, avg_loss,
																				optimizer.param_groups[0]['lr']))
		if ep % 10 == 9 and ep > 0:
			optimizer.param_groups[0]['lr'] *= 0.7
			top1acc = 0
			totTrainPoints = 0
			for idx, (images, labels) in enumerate(trainDataLoader):
				with torch.no_grad():
					feats = network.backbone(images.cuda())
					logits = face_classifier(feats)
				top, pred = utils.calc_accuracy(logits, labels.cuda(), topk = (1,))
				top1acc += top[0].item() * pred.shape[0]
				totTrainPoints += pred.shape[0]
			top1acc /= totTrainPoints

			print("Train Accuracies 1:", top1acc)

			top1acc = 0
			top2acc = 0
			totTestPoints = 0
			for _, (images, labels) in enumerate(testDataLoader):
				with torch.no_grad():
					feats = network.backbone(images.cuda())
					logits = face_classifier(feats)
				top, pred = utils.calc_accuracy(logits, labels.cuda(), topk = (1, 5))
				top1acc += top[0].item() * pred.shape[0]
				top2acc += top[1].item() * pred.shape[0]
				totTestPoints += pred.shape[0]
			top1acc /= totTestPoints
			top2acc /= totTestPoints

			print("Test Accuracies 1 and 5:", top1acc, top2acc)

	top1acc = 0
	totTrainPoints = 0
	for idx, (images, labels) in enumerate(trainDataLoader):
		with torch.no_grad():
			feats = network.backbone(images.cuda())
			logits = face_classifier(feats)
		top, pred = utils.calc_accuracy(logits, labels.cuda(), topk = (1,))
		top1acc += top[0].item() * pred.shape[0]
		totTrainPoints += pred.shape[0]
	top1acc /= totTrainPoints

	print("Train Accuracies 1:", top1acc)

	top1acc = 0
	top2acc = 0
	totTestPoints = 0
	for _, (images, labels) in enumerate(testDataLoader):
		with torch.no_grad():
			feats = network.backbone(images.cuda())
			logits = face_classifier(feats)
		top, pred = utils.calc_accuracy(logits, labels.cuda(), topk = (1, 5))
		top1acc += top[0].item() * pred.shape[0]
		top2acc += top[1].item() * pred.shape[0]
		totTestPoints += pred.shape[0]
	top1acc /= totTestPoints
	top2acc /= totTestPoints

	print("Test Accuracies 1 and 5:", top1acc, top2acc)



