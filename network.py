import torchvision.models as models
import torch.nn as nn


class SimClRNet(nn.Module):

	def __init__(self, numClasses):
		super().__init__()
		self.backbone = models.resnet18(pretrained = False, num_classes = 256)
		feat_num = self.backbone.fc.in_features
		self.fc = nn.Sequential(nn.Linear(feat_num, feat_num), nn.ReLU(inplace = True), nn.Linear(feat_num, 1024))
		self.backbone.fc = nn.Identity()
		self.feat_num = feat_num
		self.classifier_fc = nn.Linear(self.feat_num, numClasses)

	def forward(self, x):
		y = self.backbone(x)
		y = self.fc(y)
		return y

	def classify(self, x):
		y = self.backbone(x)
		y = self.classifier_fc(y)
		return y

	def freeze_feat(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True
		self.backbone.eval()
		self.fc.eval()
		self.classifier_fc.train()
		print("Freezing backbone and head.")

	def freeze_classifier(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
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
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True

