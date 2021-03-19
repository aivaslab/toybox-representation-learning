import argparse


def get_parser(desc):
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument('--batch-size', '-b', default = 64, type = int, help = "Batch Size for experiments")
	parser.add_argument('--epochs1', '-e1', default = 100, type = int, help = "Number of epochs for unsupervised training")
	parser.add_argument('--epochs2', '-e2', default = 50, type = int, help = "Number of epochs of supervised training")
	parser.add_argument('--resume', '-r', default = False, action = 'store_true', help = "Ignore if training from scratch"
																						 ", use if continuing training")
	parser.add_argument('--resumeFile', '-rf', default = "", type = str)
	parser.add_argument('--seed', '-s', default = -1, type = int, help = "Seed for training")
	parser.add_argument('--distort', '-d', choices = ['self', 'object', 'transform'], required = True, help = "Choose "
								"distortion for images. self for augmentations, object for other view of same object, "
								"transform for other image of object from same video")
	parser.add_argument('--lr', '-lr', default = 0.08, type = float, help = "Learning rate for unsupervised learning")
	parser.add_argument('--lr-ft', '-lr_ft', default = 0.005, type = float, help = "Learning rate for supervised "
																				   "training.")
	parser.add_argument('--save', '-sv', default = False, action = "store_true", help = "Use to save trained models.")
	parser.add_argument('--adj', '-a', default = -1, type = int)
	parser.add_argument('--frac', '-f', default = 0.1, type = float)
	parser.add_argument('--freeze-backbone', '-fb', default = True, action = 'store_false')
	parser.add_argument('--hypertune', '-ht', default = False, action = 'store_true')
	parser.add_argument('--saveName', '-sn', default = "", type = str)
	parser.add_argument('--saveRate', '-sr', default = -1, type = int)
	parser.add_argument('--transform', '-tr', default = 1, type = int)
	return parser.parse_args()
