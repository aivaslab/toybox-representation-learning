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

mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)


def get_parser(desc):
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument('--model', '-m', required = True, type = str)
	parser.add_argument('--lr', '-lr', required = True, type = float)
	parser.add_argument('--epochs', '-e', required = True, type = int)

	return parser.parse_args()


class data_face(torch.utils.data.Dataset):

	def __init__(self, train = True, transform = None):
		self.train = train
		self.transform = transform
		if self.train:
			self.data = pickle.load(open("./data2/toybox_face_train.pickle", "rb"))
			self.targets = list(csv.DictReader(open("./data2/toybox_face_train.csv", "r")))
		else:
			self.data = pickle.load(open("./data2/toybox_face_test.pickle", "rb"))
			self.targets = list(csv.DictReader(open("./data2/toybox_face_test.csv", "r")))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		im = cv2.imdecode(self.data[index], 3)
		target = int(self.targets[index]['Label'])
		if self.transform is not None:
			im = self.transform(im)

		return index, im, target


if __name__ == "__main__":
	network = comb_network.SimClRNet(numClasses = 12).cuda()
	# network = simclr_net.SimClRNet(numClasses = 12).cuda()
	args = vars(get_parser("Face Learner"))

	fileName = "./pose_models/" + args['model']
	# assert(os.path.isfile(fileName))
	network.load_state_dict(torch.load(fileName))
	network.freeze_all_params()
	featSize = network.classifier_fc.in_features
	face_classifier = nn.Sequential(nn.Linear(featSize, featSize//2), nn.ReLU(), nn.Linear(featSize//2, 6)).cuda()

	face_data_train = data_face(train = True, transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
																			  transforms.Normalize(mean, std)]))
	trainDataLoader = torch.utils.data.DataLoader(face_data_train, batch_size = 64, shuffle = True,
													  num_workers = 4)

	face_data_test = data_face(train = False, transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
																			  transforms.Normalize(mean, std)]))
	testDataLoader = torch.utils.data.DataLoader(face_data_test, batch_size = 64, shuffle = True,
													  num_workers = 4)
	optimizer = torch.optim.SGD(face_classifier.parameters(), lr = args['lr'], weight_decay = 1e-4)
	# optimizer.add_param_group({'params': network.backbone.parameters()})

	numEpochs = args['epochs']
	for ep in range(numEpochs):
		avg_loss = 0
		b = 0
		tqdmBar = tqdm.tqdm(trainDataLoader)
		for idx, images, targets in tqdmBar:
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
			for _, (indices, images, labels) in enumerate(trainDataLoader):
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
			for _, (indices, images, labels) in enumerate(testDataLoader):
				with torch.no_grad():
					feats = network.backbone(images.cuda())
					logits = face_classifier(feats)
				top, pred = utils.calc_accuracy(logits, labels.cuda(), topk = (1, 2))
				top1acc += top[0].item() * indices.size()[0]
				top2acc += top[1].item() * pred.shape[0]
				totTestPoints += indices.size()[0]
			top1acc /= totTestPoints
			top2acc /= totTestPoints

			print("Test Accuracies 1 and 2:", top1acc, top2acc)



