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


classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']

TEST_NO = {
	'ball'      : [1, 7, 9],
	'spoon'     : [5, 7, 8],
	'mug'       : [12, 13, 14],
	'cup'       : [12, 13, 15],
	'giraffe'   : [1, 5, 13],
	'horse'     : [1, 10, 15],
	'cat'       : [4, 9, 15],
	'duck'      : [5, 9, 13],
	'helicopter': [5, 10, 15],
	'airplane'  : [2, 6, 15],
	'truck'     : [2, 6, 8],
	'car'       : [6, 11, 13],
}

VAL_NO = {
	'airplane': [30, 29, 28],
	'ball': [30, 29, 28],
	'car': [30, 29, 28],
	'cat': [30, 29, 28],
	'cup': [30, 29, 28],
	'duck': [30, 29, 28],
	'giraffe': [30, 29, 28],
	'helicopter': [30, 29, 28],
	'horse': [30, 29, 28],
	'mug': [30, 29, 28],
	'spoon': [30, 29, 28],
	'truck': [30, 29, 28]
}


class data_simclr(torch.utils.data.Dataset):

	def __init__(self, root, rng, train = True, transform = None, nViews = 2, size = 224, split =
				"unsupervised", fraction = 1.0, distort = 'self', adj = -1, hyperTune = True, frac_by_object = False):

		self.train = train
		self.root = root
		self.transform = transform
		self.nViews = nViews
		self.size = size
		self.split = split
		self.fraction = fraction
		self.distort = distort
		self.adj = adj
		self.hyperTune = hyperTune
		self.rng = rng
		self.objectsSelected = None
		if not self.hyperTune:
			self.trainImagesFile = "./data/toybox_data_cropped_train.pickle"
			self.trainLabelsFile = "./data/toybox_data_cropped_train.csv"
			self.testImagesFile = "./data/toybox_data_cropped_test.pickle"
			self.testLabelsFile = "./data/toybox_data_cropped_test.csv"
		else:
			self.trainImagesFile = "./data/toybox_data_cropped_dev.pickle"
			self.trainLabelsFile = "./data/toybox_data_cropped_dev.csv"
			self.testImagesFile = "./data/toybox_data_cropped_val.pickle"
			self.testLabelsFile = "./data/toybox_data_cropped_val.csv"

		super().__init__()
		assert(distort == 'self' or distort == 'object' or distort == 'transform')
		if self.split == "unsupervised":
			if self.train:
				with open(self.trainImagesFile, "rb") as pickleFile:
					self.train_data = pickle.load(pickleFile)
				with open(self.trainLabelsFile, "r") as csvFile:
					self.train_csvFile = list(csv.DictReader(csvFile))
				if frac_by_object:
					self.indicesSelected = self.select_indices_object()
				else:
					lenWholeData = len(self.train_data)
					lenTrainData = int(self.fraction * lenWholeData)
					self.indicesSelected = rng.choice(lenWholeData, lenTrainData, replace = False)
			else:
				with open(self.testImagesFile, "rb") as pickleFile:
					self.test_data = pickle.load(pickleFile)
		else:
			if self.train:
				with open(self.trainImagesFile, "rb") as pickleFile:
					self.train_data = pickle.load(pickleFile)
				with open(self.trainLabelsFile, "r") as csvFile:
					self.train_csvFile = list(csv.DictReader(csvFile))
				if frac_by_object:
					self.indicesSelected = self.select_indices_object()
				else:
					lenWholeData = len(self.train_data)
					lenTrainData = int(self.fraction * lenWholeData)
					self.indicesSelected = rng.choice(lenWholeData, lenTrainData, replace = False)
			else:
				with open(self.testImagesFile, "rb") as pickleFile:
					self.test_data = pickle.load(pickleFile)
				with open(self.testLabelsFile, "r") as csvFile:
					self.test_csvFile = list(csv.DictReader(csvFile))


	def select_indices_object(self):
		numObjectsPerClassTrain = 27 - 3 * self.hyperTune
		numObjectsPerClassSelected = math.ceil(self.fraction * numObjectsPerClassTrain)
		objectsSelected = {}
		for cl in range(len(classes)):
			objectsInTrain = []
			for i in range(30):
				if i not in TEST_NO[classes[cl]]:
					if self.hyperTune:
						if i not in VAL_NO[classes[cl]]:
							objectsInTrain.append(i)
					else:
						objectsInTrain.append(i)
			print(cl, objectsInTrain)
			objectsSel = self.rng.choice(objectsInTrain, numObjectsPerClassSelected)
			for obj in objectsSel:
				assert(obj not in TEST_NO[classes[cl]])
				if self.hyperTune:
					assert(obj not in VAL_NO[classes[cl]])
			print(objectsSel)
			objectsSelected[cl] = objectsSel
		self.objectsSelected = objectsSelected
		indicesSelected = []
		with open(self.trainLabelsFile, "r") as csvFile:
			train_csvFile = list(csv.DictReader(csvFile))
		for i in range(len(train_csvFile)):
			cl, obj = train_csvFile[i]['Class ID'], train_csvFile[i]['Object']
			if int(obj) in objectsSelected[int(cl)]:
				indicesSelected.append(i)

		return indicesSelected

	def __len__(self):
		if self.train:
			return len(self.indicesSelected)
		else:
			return len(self.test_data)

	def __getitem__(self, index):

		if self.train:
			actualIndex = self.indicesSelected[index]
			img = np.array(cv2.imdecode(self.train_data[actualIndex], 3))
			if self.split == "unsupervised":
				label = -1
			else:
				label = self.train_csvFile[actualIndex]['Class ID']
		else:
			actualIndex = index
			img = np.array(cv2.imdecode(self.test_data[index], 3))
			label = self.test_csvFile[index]['Class ID']

		if self.split == "unsupervised":
			if self.distort == 'self':
				if self.transform is not None:
					imgs = [self.transform(img) for _ in range(self.nViews)]
				else:
					imgs = [img, img]

			elif self.distort == 'object':
				low, high = int(self.train_csvFile[actualIndex]['Obj Start']), int(self.train_csvFile[actualIndex]['Obj End'])
				id2 = self.rng.integers(low = low, high = high + 1, size = 1)[0]
				img2 = np.array(cv2.imdecode(self.train_data[id2], 3))
				if self.transform is not None:
					imgs = [self.transform(img), self.transform(img2)]
				else:
					imgs = [img, img2]
			else:
				if self.adj == -1:
					low, high = int(self.train_csvFile[actualIndex]['Tr Start']), int(
						self.train_csvFile[actualIndex]['Tr End'])
					id2 = self.rng.randint(low, high)
				else:
					low = max(0, actualIndex - self.adj)
					high = min(int(len(self.train_data) * self.fraction) - 1, actualIndex + self.adj)
					try:
						if self.train_csvFile[low]['Transformation'] != self.train_csvFile[actualIndex]['Transformation']:
							id2 = high
						elif self.train_csvFile[high]['Transformation'] != self.train_csvFile[actualIndex]['Transformation']:
							id2 = low
						else:
							id2 = self.rng.choice([low, high])
					except IndexError:
						print(low, actualIndex, high)
				# print(actualIndex, id2, self.train_csvFile[actualIndex]['Transformation'] ==
				#	  self.train_csvFile[id2]['Transformation'])
				img2 = np.array(cv2.imdecode(self.train_data[id2], 3))
				if self.transform is not None:
					imgs = [self.transform(img), self.transform(img2)]
				else:
					imgs = [img, img2]
		else:
			if self.transform is not None:
				imgs = self.transform(img)
			else:
				imgs = img
		return actualIndex, imgs, int(label)


def online_mean_and_sd(loader):
	"""Compute the mean and sd in an online fashion

		Var[x] = E[X^2] - E^2[X]
	"""
	cnt = 0
	fst_moment = torch.empty(3)
	snd_moment = torch.empty(3)

	for _, (data, _), _ in loader:
		print(data.shape)
		b, c, h, w = data.shape
		nb_pixels = b * h * w
		sum_ = torch.sum(data, dim=[0, 2, 3])
		sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
		fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
		snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

		cnt += nb_pixels

	return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


if __name__ == "__main__":
	rng = np.random.default_rng(5)
	simclr = data_simclr(root = "./data", rng = rng, train = True, nViews = 2, size = 224,
								transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]), fraction = 0.1,
						 distort = "self", adj = -1, hyperTune = True, frac_by_object = True)
	trainDataLoader = torch.utils.data.DataLoader(simclr, batch_size = 64, shuffle = True,
													  num_workers = 2)

	print(len(simclr))


	# mean, std = online_mean_and_sd(trainDataLoader)
	# print(mean, std)
