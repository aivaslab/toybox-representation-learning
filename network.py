import torchvision.models as models
import torch.nn as nn
import copy


def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)


class SimClRNet(nn.Module):

	def __init__(self, numClasses):
		super().__init__()
		self.backbone = models.resnet18(pretrained = False, num_classes = 256)
		feat_num = self.backbone.fc.in_features
		self.fc = nn.Sequential(nn.Linear(feat_num, 2*feat_num), nn.ReLU(inplace = True), nn.Linear(2*feat_num, 128))
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
		print("Freezing backbone and unsupervised head.")

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


class BYOLNet(nn.Module):

	def __init__(self, numClasses, beta = 0.999):
		super().__init__()
		self.numClasses = numClasses
		self.encoder_backbone = models.resnet18(pretrained = False, num_classes = 256)
		self.num_features = self.encoder_backbone.fc.in_features
		self.beta = beta
		feat_num = self.num_features
		self.encoder_backbone.fc = nn.Identity()
		self.encoder_projection = nn.Sequential(nn.Linear(in_features = feat_num, out_features = 2*feat_num),
												nn.BatchNorm1d(num_features = 2*feat_num), nn.ReLU(inplace = True),
												nn.Linear(in_features = 2*feat_num, out_features = 128))
		self.encoder_prediction = nn.Sequential(nn.Linear(in_features = 128, out_features = feat_num),
												nn.BatchNorm1d(num_features = feat_num), nn.ReLU(inplace = True),
												nn.Linear(in_features = feat_num, out_features = 128))

		self.classifier_fc = nn.Linear(feat_num, numClasses)
		self.encoder_backbone.apply(init_weights)
		self.encoder_projection.apply(init_weights)
		self.encoder_prediction.apply(init_weights)
		self.classifier_fc.apply(init_weights)
		self.target_backbone = copy.deepcopy(self.encoder_backbone)
		self.target_projection = copy.deepcopy(self.encoder_projection)

	def update_target_network(self):
		for current_params, ma_params in zip(self.encoder_backbone.parameters(), self.target_backbone.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = old_weight * self.beta + up_weight * (1 - self.beta)

		for current_params, ma_params in zip(self.encoder_projection.parameters(), self.target_projection.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = old_weight * self.beta + up_weight * (1 - self.beta)

	def encoder_forward(self, x):
		y = self.encoder_backbone(x)
		# print(y.shape)
		y = self.encoder_projection(y)
		y = self.encoder_prediction(y)

		return y

	def target_forward(self, x):
		y = self.target_backbone(x)
		y = self.target_projection(y)

		return y

	def classify(self, x):
		y = self.encoder_backbone(x)
		y = self.classifier_fc(y)

		return y

	def freeze_encoder_backbone(self):
		for name, param in self.encoder_backbone.named_parameters():
			param.requires_grad = False
		# self.encoder_backbone.eval()
		print("Freezing encoder network backbone.....")

	def unfreeze_encoder_backbone(self):
		for name, param in self.encoder_backbone.named_parameters():
			param.requires_grad = True
		# self.encoder_backbone.train()
		print("Unfreezing encoder network backbone.....")

	def freeze_encoder_projection(self):
		for name, param in self.encoder_projection.named_parameters():
			param.requires_grad = False
		# self.encoder_projection.eval()
		print("Freezing encoder network projection.....")

	def unfreeze_encoder_projection(self):
		for name, param in self.encoder_projection.named_parameters():
			param.requires_grad = True
		# self.encoder_projection.train()
		print("Unfreezing encoder network projection.....")

	def freeze_encoder_prediction(self):
		for name, param in self.encoder_prediction.named_parameters():
			param.requires_grad = False
		# self.encoder_prediction.eval()
		print("Freezing encoder network prediction.....")

	def unfreeze_encoder_prediction(self):
		for name, param in self.encoder_prediction.named_parameters():
			param.requires_grad = True
		# self.encoder_prediction.train()
		print("Unfreezing encoder network prediction.....")

	def freeze_target_backbone(self):
		for name, param in self.target_backbone.named_parameters():
			param.requires_grad = False
		# self.target_backbone.eval()
		print("Freezing target network backbone.....")

	def unfreeze_target_backbone(self):
		for name, param in self.target_backbone.named_parameters():
			param.requires_grad = True
		# self.target_backbone.train()
		print("Unfreezing target network backbone.....")

	def freeze_target_projection(self):
		for name, param in self.target_projection.named_parameters():
			param.requires_grad = False
		# self.target_projection.eval()
		print("Freezing target network projection.....")

	def unfreeze_target_projection(self):
		for name, param in self.target_projection.named_parameters():
			param.requires_grad = True
		# self.target_projection.train()
		print("Unfreezing target network projection.....")

	def freeze_classifier(self):
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
		# self.classifier_fc.eval()
		print("Freezing classifier fc.....")

	def unfreeze_classifier(self):
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True
		# self.classifier_fc.train()
		print("Unfreezing classififer fc.....")




