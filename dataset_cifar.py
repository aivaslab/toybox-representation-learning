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


class data_cifar10(datasets.CIFAR10):

	def __init__(self, root, rng, train = True, transform = None, nViews = 2, size = 224, split =
				"unsupervised", fraction = 1.0, distort = 'self', adj = -1, hyperTune = True, distortArg = False):

		if hyperTune:
			super().__init__(root = root, train = True, download = True)
		else:
			super().__init__(root = root, train = train, download = True)

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
		self.distortArg = distortArg

		if self.hyperTune:
			if self.train:
				range_low = 0
				range_high = int(0.8 * len(self.data))
			else:
				range_low = int(0.8 * len(self.data))
				range_high = len(self.data)
		else:
			range_low = 0
			range_high = len(self.data)

		arr = np.arange(range_low, range_high)
		print("Split:", self.train, np.min(arr), np.max(arr))
		len_data = range_high - range_low

		indices = self.rng.choice(arr, size = int(fraction * len_data), replace = False)

		unique = len(indices) == len(set(indices))
		assert unique
		assert len(indices) == int(fraction * len_data)

		if self.train:
			self.train_data = self.data[indices]
			self.train_labels = np.array(self.targets)[indices]
		else:
			self.test_data = self.data[indices]
			self.test_labels = np.array(self.targets)[indices]

	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.test_data)

	def __getitem__(self, index):

		if self.train:
			img = self.train_data[index]
			if self.split == "unsupervised":
				label = -1
			else:
				label = self.train_labels[index]
		else:
			img = self.test_data[index]
			label = self.test_labels[index]

		if self.split == "unsupervised":
			if self.transform is not None:
				imgs = [self.transform(img) for _ in range(self.nViews)]
			else:
				imgs = [img, img]
		else:
			if self.transform is not None:
				imgs = self.transform(img)
			else:
				imgs = img

		return index, imgs, int(label)
