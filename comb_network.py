import torchvision.models as models
import torch.nn as nn
import copy


def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)


class Identity(nn.Module):

	def __init__(self):
		super(Identity, self).__init__()

	def __call__(self, x):
		return x


class SimClRNet(nn.Module):

	def __init__(self, numClasses):
		super().__init__()
		self.backbone = models.resnet18(pretrained = False, num_classes = 256)
		feat_num = self.backbone.fc.in_features
		self.fc = nn.Sequential(nn.Linear(feat_num, feat_num), nn.ReLU(inplace = True), nn.Linear(feat_num, 128))
		self.fc2 = nn.Sequential(nn.Linear(feat_num, feat_num//2), nn.ReLU(inplace = True), nn.Linear(feat_num//2, 16))
		self.backbone.fc = Identity()
		self.feat_num = feat_num
		self.classifier_fc = nn.Linear(self.feat_num, numClasses)
		self.unsupervised = True

	def forward(self, x):
		y = self.backbone(x)
		if self.unsupervised:
			z1 = self.fc(y)
			z2 = self.fc2(y)
			return z1, z2
		else:
			y = self.classifier_fc(y)
			return y

	def freeze_feat(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.fc2.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True
		self.backbone.eval()
		self.fc.eval()
		self.classifier_fc.train()
		print("Freezing backbone and unsupervised head.")

	def freeze_classifier(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = True
		for name, param in self.fc2.named_parameters():
			param.requires_grad = True
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
		self.backbone.train()
		self.fc.train()
		self.classifier_fc.eval()
		print("Freezing classifier fc.")

	def freeze_head(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.fc2.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True
		self.backbone.train()
		self.fc.eval()
		self.classifier_fc.train()
		print("Freezing only unsupervised head fc.")

	def unfreeze_all_params(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = True
		for name, param in self.fc2.named_parameters():
			param.requires_grad = True
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True

	def freeze_all_params(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.fc2.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
