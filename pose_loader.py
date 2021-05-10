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


class pose_data(torch.utils.data.Dataset):

	def __init__(self, root, rng, train = True, transform = None, nViews = 2, size = 224, split =
				"unsupervised", fraction = 1.0, hyperTune = True, frac_by_object = False,
				 distortArg = False):

		self.train = train
		self.root = root
		self.transform = transform
		self.nViews = nViews
		self.size = size
		self.split = split
		self.fraction = fraction
		self.hyperTune = hyperTune
		self.rng = rng
		self.objectsSelected = None
		self.distortArg = distortArg

		super().__init__()
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
			gap1 = 4
			gap2 = 8

			low1 = max(0, actualIndex - gap1)
			high1 = min(int(len(self.train_data)) - 1, actualIndex + gap1)
			try:
				if self.train_csvFile[low1][self.tr_key] != self.train_csvFile[actualIndex][self.tr_key]:
					low1 = actualIndex
				elif self.train_csvFile[high1][self.tr_key] != self.train_csvFile[actualIndex][self.tr_key]:
					high1 = actualIndex
				if self.distortArg:
					id2 = self.rng.choice([low1, high1], 1)[0]
				else:
					id2 = self.rng.integers(low = low1, high = high1 + 1, size = 1)[0]
			except IndexError:
				print(low1, actualIndex, high1)
			low22 = min(int(len(self.train_data)) - 1, actualIndex + gap1 + 1)
			high22 = min(int(len(self.train_data)) - 1, actualIndex + gap2)
			high21 = max(0, actualIndex - gap1 - 1)
			low21 = max(0, actualIndex - gap2)
			if id2 < actualIndex:
				id3 = self.rng.choice([low21, high21], 1)[0]
			else:
				id3 = self.rng.choice([low22, high22], 1)[0]

			img2 = np.array(cv2.imdecode(self.train_data[id2], 3))
			img3 = np.array(cv2.imdecode(self.train_data[id3], 3))
			# print(actualIndex, id2, id3)
			if self.transform is not None:
				imgs = [self.transform(img), self.transform(img2), self.transform(img3)]
			else:
				imgs = [img, img2]
		else:
			if self.transform is not None:
				imgs = self.transform(img)
			else:
				imgs = img
		return actualIndex, imgs, int(label)


