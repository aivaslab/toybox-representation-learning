import pickle

fileName = "./output/trial_run_config.pickle"
file = open(fileName, "rb")
ff = pickle.load(file)
print(ff)

fileName = "./output/trial_train_indices.pickle"
file = open(fileName, "rb")
ff = pickle.load(file)
print(len(ff))

fileName = "./output/trial_run_train_losses.pickle"
file = open(fileName, "rb")
ff = pickle.load(file)
print(len(ff))