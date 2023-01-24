"""Module for visualizing the first convolution layer filter images"""
import torch
import network
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "../filter_images/"


def generate_filters(model_path, out_file_name):
	"""Generate the first layer filters"""
	net = network.SimClRNet(num_classes=12)
	net.load_state_dict(torch.load(model_path))
	model_children = list(net.backbone.children())
	print(model_children[0], type(model_children[0]), model_children[0].weight.shape)
	convlayer1 = model_children[0]

	# visualize the first conv layer filters
	plt.figure(figsize=(20, 17), tight_layout=True)
	for i, fltr in enumerate(model_children[0].weight):
		plt.subplot(8, 8, i+1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
		fltr = fltr.detach().numpy()
		fltr = fltr - fltr.min()
		fltr = fltr / fltr.max()
		fltr = np.transpose(fltr, (1, 2, 0))
		plt.imshow(fltr)
		plt.axis('off')
		plt.savefig(OUT_PATH + out_file_name)
	# plt.show()


generate_filters("../simclr_models/simclr_toybox_self_final_1_supervised.pt", "self_1.png")
generate_filters("../simclr_models/simclr_toybox_self_final_2_supervised.pt", "self_2.png")
generate_filters("../simclr_models/simclr_toybox_transform_final_1_supervised.pt", "transform_1.png")
generate_filters("../simclr_models/simclr_toybox_transform_final_2_supervised.pt", "transform_2.png")
generate_filters("../simclr_models/simclr_toybox_object_final_1_supervised.pt", "object_1.png")
generate_filters("../simclr_models/simclr_toybox_object_final_2_supervised.pt", "object_2.png")
generate_filters("../simclr_models/toybox_class_final_supervised.pt", "class_1.png")
generate_filters("../simclr_models/toybox_class_final_2_supervised.pt", "class_2.png")
