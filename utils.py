import torch
import torch.nn.functional as F
import torch.nn


class UnNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""
		for t, m, s in zip(tensor, self.mean, self.std):
			t.mul_(s).add_(m)
			# The normalize code -> t.sub_(m).div_(s)
		return tensor


class TripletMarginWithDistanceLoss:
	def __init__(self, distanceFunction = None, margin = 0.5):
		self.distanceFunction = distanceFunction
		self.margin = margin

	def __call__(self, anchor, positive, negative):
		if self.distanceFunction is None:
			positive_dist = torch.pairwise_distance(anchor, positive)
			negative_dist = torch.pairwise_distance(anchor, negative)
		else:
			positive_dist = self.distanceFunction(anchor, positive)
			negative_dist = self.distanceFunction(anchor, negative)
		# print(positive_dist.shape, negative_dist.shape)
		return torch.clamp(positive_dist - negative_dist + self.margin, min = 0.0).mean()


def info_nce_loss(features, temp):
	dev = torch.device('cuda:0')
	batchSize = features.shape[0] / 2
	labels = torch.cat([torch.arange(batchSize) for _ in range(2)], dim = 0)
	labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
	labels = labels.to(dev)

	features = F.normalize(features, dim = 1)

	similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1))
	# discard the main diagonal from both: labels and similarities matrix
	mask = torch.eye(labels.shape[0], dtype = torch.bool).to(dev)
	labels = labels[~mask].view(labels.shape[0], -1).type(torch.uint8)
	# print(labels)
	similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
	positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
	negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

	logits = torch.cat([positives, negatives], dim = 1)
	labels = torch.zeros(logits.shape[0], dtype = torch.long).to(dev)

	logits = logits / temp
	return logits, labels


def calc_accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batchSize = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		target_reshaped = torch.reshape(target, (1, -1)).repeat(maxk, 1)
		correct_top_k = torch.eq(pred, target_reshaped)
		pred_1 = pred[0]
		res = []
		for k in topk:
			correct_k = correct_top_k[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(torch.mul(correct_k, 100.0 / batchSize))
		return res, pred_1
