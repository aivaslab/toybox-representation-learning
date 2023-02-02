"""Module for supervised dataloader"""
import cv2
import numpy as np
import csv
import pickle
import os
import torch.utils.data

mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)


class data_loader(torch.utils.data.Dataset):
    """Data loader class"""
    def __init__(self, root, rng, train=True, transform=None, size=224, hypertune=False, fraction=1.0,
                 split=None):
        
        self.train = train
        self.root = root
        self.transform = transform
        self.hypertune = hypertune
        self.size = size
        self.fraction = fraction
        self.rng = rng
        self.data_path = self.root
        self.split = split
        try:
            assert os.path.isdir(self.data_path)
        except AssertionError:
            assert isinstance(self.data_path, object)
            raise AssertionError("Data directory not found:", self.data_path)
        self.label_key = 'Class ID'
        if self.hypertune:
            self.trainImagesFile = self.data_path + "toybox_data_cropped_dev.pickle"
            self.trainLabelsFile = self.data_path + "toybox_data_cropped_dev.csv"
            self.testImagesFile = self.data_path + "toybox_data_cropped_val.pickle"
            self.testLabelsFile = self.data_path + "toybox_data_cropped_val.csv"
        else:
            self.trainImagesFile = self.data_path + "toybox_data_cropped_train.pickle"
            self.trainLabelsFile = self.data_path + "toybox_data_cropped_train.csv"
            self.testImagesFile = self.data_path + "toybox_data_cropped_test.pickle"
            self.testLabelsFile = self.data_path + "toybox_data_cropped_test.csv"
        
        super().__init__()
        
        if self.train:
            with open(self.trainImagesFile, "rb") as pickleFile:
                self.train_data = pickle.load(pickleFile)
            with open(self.trainLabelsFile, "r") as csvFile:
                self.train_csvFile = list(csv.DictReader(csvFile))
            lenWholeData = len(self.train_data)
            lenTrainData = int(self.fraction * lenWholeData)
            self.indicesSelected = rng.choice(lenWholeData, lenTrainData, replace=False)
        else:
            with open(self.testImagesFile, "rb") as pickleFile:
                self.test_data = pickle.load(pickleFile)
            with open(self.testLabelsFile, "r") as csvFile:
                self.test_csvFile = list(csv.DictReader(csvFile))
    
    def __len__(self):
        if self.train:
            return len(self.indicesSelected)
        else:
            if self.split == "instance" and self.instance:
                return len(self.indicesSelected)
            else:
                return len(self.test_data)
    
    def __getitem__(self, index):
        if self.train:
            actualIndex = self.indicesSelected[index]
            img = np.array(cv2.imdecode(self.train_data[actualIndex], 3))
            label = self.train_csvFile[actualIndex][self.label_key]
        else:
            actualIndex = index
            img = np.array(cv2.imdecode(self.test_data[index], 3))
            label = self.test_csvFile[index][self.label_key]
        
        if self.transform is not None:
            imgs = self.transform(img)
        else:
            imgs = img
        return actualIndex, imgs, int(label)
