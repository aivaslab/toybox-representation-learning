import simclr
import time

start_time = time.time()

default_args = {
	'batch_size' : 64,
	'epochs1' : 10,
	'epochs2' : 10,
	'resume' : False,
	'resumeFile' : "",
	'seed' : -1,
	'distort' : "self",
	'adj' : -1,
	'lr' : 0.08,
	'lr_ft' : 0.02,
	'save' : True,
	'frac1' : 1.0,
	'frac2' : 0.1,
	'freeze_backbone' : True,
	'hypertune' : True,
	'saveName' : "temp",
	'saveRate' : -1,
	'transform' : 1,
	'temperature' : 0.1,
	'weight_decay' : 1e-6,
	'epochsRan' : -1,
	'supervisedRep' : 1,
	'dataset' : "toybox",
	'distortArg' : False,
	'workers' : 4
}

simclr.train_unsupervised_and_supervised(args = default_args)

mean_accs = []

default_args['resume'] = True
default_args['resumeFile'] = "temp_unsupervised_final.pt"
default_args['lr_ft'] = 0.02
default_args['supervisedRep'] = 5
default_args['save'] = False
default_args['saveName'] = ""
print("------------------------------------------------------------------------------------------------")
print("Starting linear evaluation for", default_args['supervisedRep'], 'reps')
mean_acc = simclr.evaluate_trained_network(args = default_args)
mean_accs.append(mean_acc)

default_args['lr_ft'] = 0.01
print("------------------------------------------------------------------------------------------------")
print("Starting linear evaluation for", default_args['supervisedRep'], 'reps')
mean_acc = simclr.evaluate_trained_network(args = default_args)
mean_accs.append(mean_acc)

default_args['lr_ft'] = 0.005
print("------------------------------------------------------------------------------------------------")
print("Starting linear evaluation for", default_args['supervisedRep'], 'reps')
mean_acc = simclr.evaluate_trained_network(args = default_args)
mean_accs.append(mean_acc)

print("The mean accuracies over the different evaluations are:", mean_accs)
print("Total time:", time.time() - start_time, "seconds")
