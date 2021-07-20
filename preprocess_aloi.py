import os
import cv2
import time
import pickle
import csv
import numpy as np

start_time = time.time()
dataPath = "../ALOI/png/"
numObjects = 10
totDegrees = 360
degStep = 5
allimages = np.arange(0, totDegrees, degStep)
totSize = len(allimages)
testSize = 12
valSize = 12
trainSize = totSize - testSize - valSize

imgFrames = []
outCSV = "./data/aloi_all_images.csv"
outCSVWriter = csv.writer(open(outCSV, "w"))
outPickle = "./data/aloi_all_images.pickle"
outPickleFile = open(outPickle, "wb")

trainImageFrames = []
trainCSV = "./data/aloi_train.csv"
trainCSVWriter = csv.writer(open(trainCSV, "w"))
trainPickle = "./data/aloi_train.pickle"
trainPickleFile = open(trainPickle, "wb")

testImageFrames = []
testCSV = "./data/aloi_test.csv"
testCSVWriter = csv.writer(open(testCSV, "w"))
testPickle = "./data/aloi_test.pickle"
testPickleFile = open(testPickle, "wb")

devImageFrames = []
devCSV = "./data/aloi_dev.csv"
devCSVWriter = csv.writer(open(devCSV, "w"))
devPickle = "./data/aloi_dev.pickle"
devPickleFile = open(devPickle, "wb")

valImageFrames = []
valCSV = "./data/aloi_val.csv"
valCSVWriter = csv.writer(open(valCSV, "w"))
valPickle = "./data/aloi_val.pickle"
valPickleFile = open(valPickle, "wb")

for i in range(1, numObjects + 1):
	dirPath = dataPath + str(i) + "//"
	if i % 10 == 0:
		print(i, time.time() - start_time)
	trainImages = np.random.choice(allimages, trainSize, replace = False)
	valImages = np.random.choice(trainImages, valSize, replace = False)
	for deg in range(0, totDegrees, degStep):
		filePath = dirPath + str(i) + "_r" + str(deg) + ".png"
		assert(os.path.isfile(filePath))
		img = cv2.imread(filePath)
		_, encoded_frame = cv2.imencode(".jpeg", img)
		imgFrames.append(encoded_frame)
		outCSVWriter.writerow([i])
		if deg in trainImages:
			trainImageFrames.append(encoded_frame)
			trainCSVWriter.writerow([i])
			if deg in valImages:
				valImageFrames.append(encoded_frame)
				valCSVWriter.writerow([i])
			else:
				devImageFrames.append(encoded_frame)
				devCSVWriter.writerow([i])
		else:
			testImageFrames.append(encoded_frame)
			testCSVWriter.writerow([i])
print(len(imgFrames), len(trainImageFrames), len(testImageFrames), len(devImageFrames), len(valImageFrames))
pickle.dump(imgFrames, outPickleFile, pickle.DEFAULT_PROTOCOL)
outPickleFile.close()

