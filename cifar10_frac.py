from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np


class fCIFAR10(CIFAR10):

	def __init__(self, root, train, download, transform, frac = 1.0):

		super(fCIFAR10, self).__init__(root = root, train = train, download = download)
		self.transform = transform
		self.frac = frac

		lenData = len(self.data)
		indices = np.random.randint(low = 0, high = lenData, size = int(frac * lenData), dtype =np.int32)
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
		return img, target


class fCIFAR100(CIFAR100):

	def __init__(self, root, train, download, transform, frac = 1.0):

		super(fCIFAR100, self).__init__(root = root, train = train, download = download)
		self.transform = transform
		self.frac = frac

		lenData = len(self.data)
		indices = np.random.randint(low = 0, high = lenData, size = int(frac * lenData), dtype =np.int32)
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
		return img, target
