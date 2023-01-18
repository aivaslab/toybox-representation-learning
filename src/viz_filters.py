import torch
import network
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
'''
resnetModel = models.resnet18(pretrained = True, num_classes = 1000)

device = torch.device('cuda:0')
# net = network.SimClRNet(numClasses = 12)
# net.load_state_dict(torch.load("trained_model_300.pt"))
model_children = list(resnetModel.children())
print(model_children[0], type(model_children[0]), model_children[0].weight.shape)
convlayer1 = model_children[0]
for i in range(5):
	print(convlayer1.weight[i].shape)
	img = transforms.ToPILImage()(convlayer1.weight[i])
	resized_img = img.resize((490, 490))
	# resized_img.show()

# visualize the first conv layer filters
plt.figure(figsize=(20, 17))
for i, fltr in enumerate(model_children[0].weight):
	plt.subplot(8, 8, i+1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
	# print(fltr * 255)
	fltr = fltr.detach().numpy()
	fltr = fltr - fltr.min()
	fltr = fltr / fltr.max()
	fltr = np.transpose(fltr, (1, 2, 0))
	print(type(fltr), fltr.shape)
	plt.imshow(fltr)
	plt.axis('off')
	plt.savefig('./filter.png')
plt.show()

'''


def generate_filters(modelName):
	modName = modelName + ".pt"
	outFileName = modelName + ".png"
	net = network.SimClRNet(num_classes= 12)
	net.load_state_dict(torch.load(modName))
	model_children = list(net.backbone.children())
	print(model_children[0], type(model_children[0]), model_children[0].weight.shape)
	convlayer1 = model_children[0]

	# visualize the first conv layer filters
	plt.figure(figsize=(20, 17))
	for i, fltr in enumerate(model_children[0].weight):
		plt.subplot(8, 8, i+1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
		# print(fltr * 255)
		fltr = fltr.detach().numpy()
		fltr = fltr - fltr.min()
		fltr = fltr / fltr.max()
		fltr = np.transpose(fltr, (1, 2, 0))
		# print(type(fltr), fltr.shape)
		plt.imshow(fltr)
		plt.axis('off')
		plt.savefig(outFileName)
	plt.show()


generate_filters("trained_model_cropped_object_-1")
