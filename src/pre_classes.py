import csv
import os


def add_classes(fileName, newFileName):
	header = ["ID", "Class", "Class ID", "Object", "Transformation", "File Name", "Obj Start", "Obj End", "Tr Start",
			  "Tr End", "CL Start", "CL End"]
	with open(fileName, "r") as csvFile:
		origcsvFile = list(csv.DictReader(csvFile))
	startIDX = {}
	endIDX = {}

	for row in origcsvFile:
		cl = int(row["Class ID"])
		idx = int(row["ID"])
		if cl not in startIDX.keys():
			startIDX[cl] = idx
			endIDX[cl] = idx
		else:
			if idx < startIDX[cl]:
				startIDX[cl] = idx
			if idx > endIDX[cl]:
				endIDX[cl] = idx

	csvFile.close()
	with open(fileName, "r") as csvFile:
		origcsvFile = list(csv.reader(csvFile))
	newFile = open(newFileName, "w")
	writer = csv.writer(newFile)
	writer.writerow(header)
	for ii in range(1, len(origcsvFile)):
		row = origcsvFile[ii]
		cl = int(row[header.index("Class ID")])
		writer.writerow(row + [startIDX[cl], endIDX[cl]])
	newFile.close()
	print(startIDX)
	print(endIDX)


trainOrig = "./data/toybox_data_cropped_train_old.csv"
testOrig = "./data/toybox_data_cropped_test_old.csv"
devOrig = "./data/toybox_data_cropped_dev_old.csv"
valOrig = "./data/toybox_data_cropped_val_old.csv"

trainOrigNew = "./data/toybox_data_cropped_train.csv"
testOrigNew = "./data/toybox_data_cropped_test.csv"
devOrigNew = "./data/toybox_data_cropped_dev.csv"
valOrigNew = "./data/toybox_data_cropped_val.csv"

assert os.path.isfile(trainOrig)
assert os.path.isfile(testOrig)
assert os.path.isfile(devOrig)
assert os.path.isfile(valOrig)


assert not os.path.isfile(trainOrigNew)
assert not os.path.isfile(testOrigNew)
assert not os.path.isfile(devOrigNew)
assert not os.path.isfile(valOrigNew)

add_classes(fileName = trainOrig, newFileName = trainOrigNew)
add_classes(fileName = testOrig, newFileName = testOrigNew)
add_classes(fileName = devOrig, newFileName = devOrigNew)
add_classes(fileName = valOrig, newFileName = valOrigNew)