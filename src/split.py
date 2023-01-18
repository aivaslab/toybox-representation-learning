import pickle
import csv
import cv2

classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']
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

VAL_NO = {
	'airplane': [30, 29, 28],
	'ball': [30, 29, 28],
	'car': [30, 29, 28],
	'cat': [30, 29, 28],
	'cup': [30, 29, 28],
	'duck': [30, 29, 28],
	'giraffe': [30, 29, 28],
	'helicopter': [30, 29, 28],
	'horse': [30, 29, 28],
	'mug': [30, 29, 28],
	'spoon': [30, 29, 28],
	'truck': [30, 29, 28]
}

trainLabelsFile = "./data2/toybox_data_interpolated_cropped_train.csv"
trainImagesFile = "./data2/toybox_data_interpolated_cropped_train.pickle"
devLabelsFile = "./data2/toybox_data_interpolated_cropped_dev.csv"
devImagesFile = "./data2/toybox_data_interpolated_cropped_dev.pickle"
valLabelsFile = "./data2/toybox_data_interpolated_cropped_val.csv"
valImagesFile = "./data2/toybox_data_interpolated_cropped_val.pickle"

dev_frames = []
val_frames = []

with open(trainImagesFile, "rb") as pickleFile:
	train_data = pickle.load(pickleFile)
with open(trainLabelsFile, "r") as csvFile:
	train_csvFile = list(csv.DictReader(csvFile))

assert(len(train_data) == len(train_csvFile))

devFile = open(devLabelsFile, "w")
devWriter = csv.writer(devFile)
devWriter.writerow(["ID", "Class", "Class ID", "Object", "Transformation", "File Name", "Obj Start", "Obj End",
						  "Tr Start", "Tr End"])
valFile = open(valLabelsFile, "w")
valWriter = csv.writer(valFile)
valWriter.writerow(["ID", "Class", "Class ID", "Object", "Transformation", "File Name", "Obj Start", "Obj End",
						  "Tr Start", "Tr End"])

devCount = 0
valCount = 0
for idx in range(len(train_data)):
	im = train_data[idx]
	cl = train_csvFile[idx]["Class"]
	obj = train_csvFile[idx]["Object"]
	tr = train_csvFile[idx]["Transformation"]
	fName = train_csvFile[idx]["File Name"]
	obj_start = train_csvFile[idx]["Obj Start"]
	obj_end = train_csvFile[idx]["Obj End"]
	tr_start = train_csvFile[idx]["Tr Start"]
	tr_end = train_csvFile[idx]["Tr End"]
	if int(obj) in VAL_NO[cl]:
		val_frames.append(im)
		valWriter.writerow([valCount, cl, classes.index(cl), obj, tr, fName, int(obj_start) - devCount, int(obj_end) -
							devCount, int(tr_start) - devCount, int(tr_end) - devCount])
		valCount += 1
	else:
		dev_frames.append(im)
		devWriter.writerow([devCount, cl, classes.index(cl), obj, tr, fName, int(obj_start) - valCount, int(obj_end) -
							valCount, int(tr_start) - valCount, int(tr_end) - valCount])
		devCount += 1

print(len(dev_frames), len(val_frames))
print(devCount, valCount)
assert(len(dev_frames) + len(val_frames) == len(train_data))

f = open(devImagesFile, 'wb')
pickle.dump(dev_frames, f, pickle.HIGHEST_PROTOCOL)
f.close()
f = open(valImagesFile, 'wb')
pickle.dump(val_frames, f, pickle.HIGHEST_PROTOCOL)
f.close()

devFile.close()
valFile.close()

