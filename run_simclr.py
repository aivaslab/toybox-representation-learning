import simclr
import time
import argparse


def get_parser(desc):
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument('--distort', '-d', choices = ['self', 'object', 'transform', 'class'], help = "Choose "
								"distortion for images. self for augmentations, object for other view of same object, "
								"transform for other image of object from same video, class for image from any other"
								"frame of same class.")
	parser.add_argument('--lr', '-lr', default = 0.08, type = float, help = "Learning rate for unsupervised learning")
	parser.add_argument('--lr-fts', '-lr_fts', nargs = '+', type = float, help = "Learning rate for supervised "
																				   "training.")
	parser.add_argument('--dataset', '-data', default = "toybox", type = str)
	parser.add_argument('--tempName', '-tn', required = True, type = str)
	parser.add_argument('--batch_size', '-b', default = 128, type = int)
	parser.add_argument('--temperature', default = 0.1, type = float)
	parser.add_argument('--epochs1', '-e1', default = 100, type = int)
	parser.add_argument('--epochs2', '-e2', default = 60, type = int, help = "Number of epochs of supervised training")

	return parser.parse_args()


start_time = time.time()
exp_args = vars(get_parser("SimCLR experiment args"))

default_args = {'batch_size': exp_args['batch_size'], 'epochs1': exp_args['epochs1'], 'epochs2': exp_args['epochs2'],
				'resume': False, 'resumeFile': "", 'seed': -1, 'distort': exp_args['distort'], 'adj': -1, 'lr': exp_args['lr'],
				'lr_ft': exp_args['lr_fts'][0],
				'save': True, 'frac1': 1.0, 'frac2': 0.1, 'freeze_backbone': True, 'hypertune': True,
				'saveName': "temp", 'saveRate': -1, 'transform' : 1, 'temperature': 0.1, 'weight_decay': 1e-6,
				'epochsRan': -1, 'supervisedRep': 1, 'dataset' : exp_args['dataset'], 'distortArg': False, 'workers': 4}

simclr.train_unsupervised_and_supervised(args = default_args)


print("------------------------------------------------------------------------------------------------")
print("Running supervised training for", len(exp_args['lr_fts']), "different learning rates")
mean_accs = []

default_args['resume'] = True
default_args['resumeFile'] = "temp_unsupervised_final.pt"
default_args['lr_ft'] = exp_args['lr_fts'][0]
default_args['supervisedRep'] = 7
default_args['save'] = False
default_args['saveName'] = ""

for rep in range(len(exp_args['lr_fts'])):
	print("Starting supervised training for lr_fts[" + str(rep + 1) + "]: " + str(exp_args['lr_fts'][rep]))
	default_args['lr_ft'] = exp_args['lr_fts'][rep]
	print("------------------------------------------------------------------------------------------------")
	print("Starting linear evaluation for", default_args['supervisedRep'], 'reps')
	mean_acc = simclr.evaluate_trained_network(args = default_args)
	print("------------------------------------------------------------------------------------------------")
	mean_accs.append(mean_acc)

print("The mean accuracies over the different evaluations are:", mean_accs)
print("Total time:", time.time() - start_time, "seconds")
