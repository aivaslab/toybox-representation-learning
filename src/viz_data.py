import csv
import pickle
import cv2

TEST_NO = {
	'ball'      : [1, 7, 9],
	'spoon'     : [5, 7, 8],
	'mug'       : [12, 13, 14],
	'cup'       : [12, 13, 15],
	'giraffe'   : [1, 5, 13],
	'horse'     : [1, 10, 15],
	'cat'       : [4, 9, 15],
	'duck'      : [5, 9, 13],
	'helicopter': [5, 10, 15],
	'airplane'  : [2, 6, 15],
	'truck'     : [2, 6, 8],
	'car'       : [6, 11, 13],
}

trainFrames = []
testFrames = []
trainLabels = csv.writer(open("./data2/toybox_face_train.csv", "w"))
testLabels = csv.writer(open("./data2/toybox_face_test.csv", "w"))
trainImages = open("./data2/toybox_face_train.pickle", "wb")
testImages = open("./data2/toybox_face_test.pickle", "wb")
classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']
trs = ['rxminus', 'rxplus', 'rzminus', 'rzplus']
trainPickle = pickle.load(open("./data2/toybox_data_interpolated_cropped_train.pickle", "rb"))
trainCSV = list(csv.DictReader(open("./data2/toybox_data_interpolated_cropped_train.csv", "r")))
testPickle = pickle.load(open("./data2/toybox_data_interpolated_cropped_test.pickle", "rb"))
testCSV = list(csv.DictReader(open("./data2/toybox_data_interpolated_cropped_test.csv", "r")))
for cl in classes:
	print(cl)
	for obj in range(1, 31):
		for tr in trs:
			if obj in TEST_NO[cl]:
				csvReader = testCSV
				imgFile = testPickle
			else:
				csvReader = trainCSV
				imgFile = trainPickle
			found = False
			for i in range(len(csvReader)):
				row = csvReader[i]
				rowcl = row['Class']
				rowobj = int(row['Object'])
				rowtr = row['Transformation']
				if cl == rowcl and obj == rowobj and rowtr == tr:
					# print(cl, obj, tr)
					found  = True
					start = int(row['Tr Start'])
					end = int(row['Tr End'])
					break
			# try:
			# 	print(start, end)
			# except NameError:
			#	print(cl, obj, tr)
			numFrames = end - start + 1
			avgGap = numFrames/8.0
			# print(avgGap)
			listFrames = [start] # , start + 1, start + 2]
			for i in range(8):
				mid = int(start + (i + 1) * avgGap)
				# listFrames.append(mid - 1)
				# listFrames.append(mid)
				if mid + 1 > end:
					listFrames.append(end)
				else:
					listFrames.append(mid + 1)
			assert(len(listFrames) == 9)
			# print(listFrames)
			if tr == "rxminus":
				targets = [0, 5, 2, 4, 0, 5, 2, 4, 0]
			elif tr == "rxplus":
				targets = [0, 4, 2, 5, 0, 4, 2, 5, 0]
			elif tr == "rzminus":
				targets = [0, 1, 2, 3, 0, 1, 2, 3, 0]
			elif tr == "rzplus":
				targets = [0, 3, 2, 1, 0, 3, 2, 1, 0]

			for i in range(9):
				try:
					if obj in TEST_NO[cl]:
						testFrames.append(imgFile[listFrames[i]])
						testLabels.writerow([targets[i]])
					else:
						trainFrames.append(imgFile[listFrames[i]])
						trainLabels.writerow([targets[i]])
					# img = cv2.imdecode(imgFile[listFrames[i]], 3)
				except IndexError:
					print(cl, obj, tr, listFrames[i])
					print(listFrames)
					raise IndexError
			# cv2.imshow("images", vconcatImg)
			# cv2.waitKey(0)
pickle.dump(trainFrames, trainImages, pickle.DEFAULT_PROTOCOL)
pickle.dump(testFrames, testImages, pickle.DEFAULT_PROTOCOL)
# trainLabels.close()
# testLabels.close()
