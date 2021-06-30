from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torch
import numpy as np


class fCIFAR10(CIFAR10):

	def __init__(self, root, train, download, transform, rng, hypertune = True, frac = 1.0):
		if hypertune:
			super(fCIFAR10, self).__init__(root = root, train = True, download = download)
		else:
			super(fCIFAR10, self).__init__(root = root, train = train, download = download)
		self.train = train
		self.transform = transform
		self.frac = frac
		self.hypertune = hypertune
		self.rng = rng

		if self.hypertune:
			if self.train:
				range_low = 0
				range_high = int(0.8 * len(self.data))
			else:
				range_low = int(0.8 * len(self.data))
				range_high = len(self.data)
		else:
			range_low = 0
			range_high = len(self.data)

		indices = self.rng.integers(low = range_low, high = range_high, size = int(frac * (range_high - range_low)),
									dtype = np.int32)
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
			return len(self.test_labels)

	def __getitem__(self, item):
		if self.train:
			img = self.train_data[item]
			target = self.train_labels[item]
		else:
			img = self.test_data[item]
			target = self.test_labels[item]
		img = self.transform(img)
		return item, img, target


class fCIFAR100(CIFAR100):

	def __init__(self, root, train, download, transform, rng, hypertune = True, frac = 1.0):

		if hypertune:
			super(fCIFAR100, self).__init__(root = root, train = True, download = download)
		else:
			super(fCIFAR100, self).__init__(root = root, train = train, download = download)
		self.train = train
		self.transform = transform
		self.frac = frac
		self.hypertune = hypertune
		self.rng = rng

		if self.hypertune:
			if self.train:
				range_low = 0
				range_high = int(0.8 * len(self.data))
			else:
				range_low = int(0.8 * len(self.data))
				range_high = len(self.data)
		else:
			range_low = 0
			range_high = len(self.data)

		indices = rng.integers(low = range_low, high = range_high, size = int(frac * (range_high - range_low)),
									dtype = np.int32)
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
			return len(self.test_labels)

	def __getitem__(self, item):
		if self.train:
			img = self.train_data[item]
			target = self.train_labels[item]
		else:
			img = self.test_data[item]
			target = self.test_labels[item]
		img = self.transform(img)
		return item, img, target


def online_mean_and_sd(loader):
	"""Compute the mean and sd in an online fashion

		Var[x] = E[X^2] - E^2[X]
	"""
	cnt = 0
	fst_moment = torch.empty(3)
	snd_moment = torch.empty(3)

	for _, data, _ in loader:
		# print(data.shape)
		b, c, h, w = data.shape
		nb_pixels = b * h * w
		sum_ = torch.sum(data, dim=[0, 2, 3])
		sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
		fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
		snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

		cnt += nb_pixels

	return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


if __name__ == "__main__":
	transform = transforms.Compose([transforms.ToPILImage(),
									# transforms.Resize(224),
									transforms.ToTensor(),
									])
	dataset = fCIFAR10(root = "./data/", train = True, download = True, transform = transform, hypertune = False,
					   frac = 1.0)
	loader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 4)
	# m, sd = online_mean_and_sd(loader = loader)
	# print(m, sd)
