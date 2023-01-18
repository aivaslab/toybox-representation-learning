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
				 distortArg = False, gap = 1):

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
		self.gap = gap
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
			gap1 = self.gap
			gap2 = 2 * self.gap

			low1 = max(0, actualIndex - gap1)
			high1 = min(int(len(self.train_data)) - 1, actualIndex + gap1)
			try:
				if self.train_csvFile[low1][self.tr_key] != self.train_csvFile[actualIndex][self.tr_key]:
					low1 = high1
				elif self.train_csvFile[high1][self.tr_key] != self.train_csvFile[actualIndex][self.tr_key]:
					high1 = low1
				# print(low1, high1)
				if self.distortArg:
					id2 = self.rng.integers(low = low1, high = high1 + 1, size = 1)[0]
				else:
					id2 = self.rng.choice([low1, high1], size = 1)[0]
			except IndexError:
				print(low1, actualIndex, high1)
			if self.distortArg:
				low22 = min(int(len(self.train_data)) - 1, id2 + 1)
				high22 = min(int(len(self.train_data)) - 1, id2 + gap1)
				high21 = max(0, id2 - 1)
				low21 = max(0, id2 - gap1)
				if id2 < actualIndex:
					id3 = self.rng.integers(low = low21, high = high21 + 1, size = 1)[0]
				else:
					id3 = self.rng.integers(low = low22, high = high22 + 1, size = 1)[0]
			else:
				if id2 < actualIndex:
					id3 = max(0, actualIndex - gap2)
					if self.train_csvFile[id3][self.tr_key] != self.train_csvFile[actualIndex][self.tr_key]:
						id3 = min(actualIndex + gap2, int(len(self.train_data)) - 1)
				else:
					id3 = min(actualIndex + gap2, int(len(self.train_data)) - 1)
					if self.train_csvFile[id3][self.tr_key] != self.train_csvFile[actualIndex][self.tr_key]:
						id3 = max(0, actualIndex - gap2)
			print(actualIndex, id2, id3, abs(id2 - actualIndex), abs(id3 - id2), abs(id3 - actualIndex))

			# img2 = np.array(cv2.imdecode(self.train_data[actualIndex], 3))
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


