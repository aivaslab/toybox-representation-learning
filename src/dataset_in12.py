"""
Dataset for IN-12
"""
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
from dataset_simclr import data_simclr

classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']


class DataIN12(torch.utils.data.Dataset):
    """
    sd
    """
    def __init__(self, root, rng, train=True, transform=None, size=224, fraction=1.0, distort='self',
                 split="unsupervised"):
        self.trainImagesFile = "../toybox_journal_experiments/data_12/IN-12/train.pickle"
        self.trainLabelsFile = "../toybox_journal_experiments/data_12/IN-12/train.csv"
        self.testImagesFile = "../toybox_journal_experiments/data_12/IN-12/test.pickle"
        self.testLabelsFile = "../toybox_journal_experiments/data_12/IN-12/test.csv"
        
        super().__init__()
        if self.train:
            with open(self.trainImagesFile, "rb") as pickleFile:
                self.train_data = pickle.load(pickleFile)
            with open(self.trainLabelsFile, "r") as csvFile:
                self.train_csvFile = list(csv.DictReader(csvFile))

            len_whole_data = len(self.train_data)
            len_train_data = int(self.fraction * len_whole_data)
            self.indicesSelected = rng.choice(len_whole_data, len_train_data, replace=False)
        else:
            with open(self.testImagesFile, "rb") as pickleFile:
                self.test_data = pickle.load(pickleFile)


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
    pass
