import torch
import copy
import time


def get_dist_mat(act_a, act_b):
	aR = torch.repeat_interleave(act_a, act_b.shape[0], dim = 0)
	bR = act_b.repeat(act_a.shape[0], 1)
	dist_mat = torch.sqrt(torch.pow(aR - bR, 2).sum(dim = 1))
	dist_mat = dist_mat.view(act_a.shape[0], -1)
	return dist_mat


def get_dist(act_a, act_b):
	start_time = time.time()
	distMat = None
	for i in range(act_a.shape[0]):
		dists = torch.sqrt(torch.pow(act_a[i] - act_b, 2).sum(dim = 1)).unsqueeze(0)
		if distMat is None:
			distMat = dists
		else:
			distMat = torch.cat((distMat, dists), dim = 0)

	print("Time taken:", time.time() - start_time)
	return distMat


a = torch.rand(size = (5000, 512)).cuda()
b = torch.rand(size = (28000, 512)).cuda()
distances = get_dist(act_a=a, act_b=b)
print(distances.shape)
dist_topk = distances.topk(k=2, dim = 1, largest = False)
