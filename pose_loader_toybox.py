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
from pose_loader import pose_data


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


class pose_data_toybox(pose_data):

	def __init__(self, root, rng, train = True, transform = None, nViews = 2, size = 224, split =
				"unsupervised", fraction = 1.0, hyperTune = True, frac_by_object = False,
				 distortArg = False):
		self.tr_start_key = 'Tr Start'
		self.tr_end_key = 'Tr End'
		self.obj_start_key = 'Obj Start'
		self.obj_end_key = 'Obj End'
		self.tr_key = 'Transformation'
		self.cl_start_key = 'CL Start'
		self.cl_end_key = 'CL End'
		if not hyperTune:
			self.trainImagesFile = "./data/toybox_data_cropped_train.pickle"
			self.trainLabelsFile = "./data/toybox_data_cropped_train.csv"
			self.testImagesFile = "./data/toybox_data_cropped_test.pickle"
			self.testLabelsFile = "./data/toybox_data_cropped_test.csv"
		else:
			self.trainImagesFile = "./data/toybox_data_cropped_dev.pickle"
			self.trainLabelsFile = "./data/toybox_data_cropped_dev.csv"
			self.testImagesFile = "./data/toybox_data_cropped_val.pickle"
			self.testLabelsFile = "./data/toybox_data_cropped_val.csv"

		super().__init__(root = root, rng = rng, train = train, transform = transform, nViews = nViews, size = size,
						 split = split, fraction = fraction, hyperTune = hyperTune,
						 frac_by_object = frac_by_object, distortArg = distortArg)


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
	simclr = pose_data_toybox(root = "./data", rng = rng, train = True, nViews = 2, size = 224,
								transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]), fraction = 1.0,
						hyperTune = True, frac_by_object = False)
	trainDataLoader = torch.utils.data.DataLoader(simclr, batch_size = 64, shuffle = True,
													  num_workers = 2)

	print(len(simclr))

	for _, (data, _, _), _ in trainDataLoader:
		pass
	# mean, std = online_mean_and_sd(trainDataLoader)
	# print(mean, std)
