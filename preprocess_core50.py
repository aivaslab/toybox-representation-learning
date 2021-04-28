import pickle
import numpy as np
import cv2
import csv
import math

dataDirectory = "./data/"

imagesFileName = dataDirectory + "core50_imgs.pickle"
labelsName = dataDirectory + "paths.pkl"
csvFile = open(dataDirectory + "test_data.csv", "w")
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["Index", "Class ID", "Object No", "Session No", "Frame No"])
images = pickle.load(open(imagesFileName, "rb"))
paths = pickle.load(open(labelsName, "rb"))
assert (len(paths) == len(images))
dicti = {}
count = 0
for i in range(len(paths)):
	# print(paths[i])
	splits = paths[i].split("_")
	# print(splits)
	sessionNum = int(splits[1])
	objNum = int(splits[2])
	imID = int(splits[3].split(".")[0])
	# print(i, sessionNum, objNum, imID)
	if (sessionNum, objNum, imID) not in dicti.keys():
		dicti[(sessionNum, objNum, imID)] = i
	else:
		count += 1
	cl_id = math.ceil(objNum/5.0)
	csvWriter.writerow([i, cl_id, objNum, sessionNum, imID])
print(count, len(dicti.keys()))
csvFile.close()
