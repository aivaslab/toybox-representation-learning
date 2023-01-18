import pickle
import csv
import numpy as np
import cv2

trainPickle = pickle.load(open("./data2/toybox_face_train.pickle", "rb"))
trainCSV = list(csv.DictReader(open("./data2/toybox_face_train.csv", "r")))

testPickle = pickle.load(open("./data2/toybox_face_test.pickle", "rb"))
testCSV = list(csv.DictReader(open("./data2/toybox_face_test.csv", "r")))

assert(len(trainCSV) == len(trainPickle))
assert(len(testCSV) == len(testPickle))

faceID = 0
for faceID in range(2):
	listIndices = []
	for i in range(len(testCSV)):
		if int(testCSV[i]['Label']) == faceID:
			listIndices.append(i)
	chosenIndices = np.random.choice(listIndices, 20, replace = False)
	print(chosenIndices)
	imgs = None
	for i in range(len(chosenIndices)):
		chosenIndex = chosenIndices[i]
		img = cv2.imdecode(testPickle[chosenIndex], 3)
		print(img.shape)
		if imgs is None:
			imgs = np.tile(img, (4, 5, 1))
		else:
			row = int(i/5)
			col = i % 5
			rowPix = row * 224
			colPix = col * 224
			imgs[rowPix : rowPix + 224, colPix : colPix + 224, :] = img
			print(row, col)
	# cv2.imshow("Image" + str(faceID), imgs)
	# cv2.waitKey(0)

	listIndices = []
	for i in range(len(trainCSV)):
		if int(trainCSV[i]['Label']) == faceID:
			listIndices.append(i)
	chosenIndices = np.random.choice(listIndices, 20, replace = False)
	print(chosenIndices)
	imgs = None
	for i in range(len(chosenIndices)):
		chosenIndex = chosenIndices[i]
		img = cv2.imdecode(trainPickle[chosenIndex], 3)
		print(img.shape)
		if imgs is None:
			imgs = np.tile(img, (4, 5, 1))
		else:
			row = int(i/5)
			col = i % 5
			rowPix = row * 224
			colPix = col * 224
			imgs[rowPix : rowPix + 224, colPix : colPix + 224, :] = img
			print(row, col)
	#cv2.imshow("Image" + str(faceID), imgs)
	#cv2.waitKey(0)
	cv2.imwrite("face" + str(faceID) + ".png", imgs)
