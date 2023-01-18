import csv
import os
import pickle
import time
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

size = 224
show = False


def crop_image(imagePath, image, l, t, w, h):
	if h < w:
		t_new = max(t - ((w - h) // 2), 0)
		h_new = min(image.shape[0], w)
		l_new = l
		w_new = w
		b_new = t_new + h_new
		if b_new > image.shape[0]:
			t_new = t_new - (b_new - image.shape[0])
	elif w < h:
		t_new = t
		h_new = h
		l_new = max(l - ((h - w) // 2), 0)
		w_new = min(image.shape[1], h)
		r_new = l_new + w_new
		if r_new > image.shape[1]:
			l_new = l_new - (r_new - image.shape[1])
	else:
		t_new = t
		h_new = h
		l_new = l
		w_new = h

	try:
		image_cropped = image[t_new:t_new + h_new, l_new:l_new + w_new]
	except ValueError:
		print(l, t, w, h)
		return None
	try:
		assert ((image_cropped.shape[1] == image_cropped.shape[0]) or w > image.shape[0])
	except AssertionError:
		print(imagePath, l, w, t, h, l_new, w_new, t_new, h_new, image_cropped.shape[1], image_cropped.shape[0])
	if show:
		cv2.imshow("image", image)
		cv2.imshow("image cropped", image_cropped)
		cv2.waitKey(0)
	return image_cropped


def generate_data(classesList, numObjects, cropFilePaths, imageDir, cropImage, trainPickle, trainCSV,
				  testPickle, testCSV):
	curr = time.time()
	trainFile = open(trainCSV, "w")
	testFile = open(testCSV, "w")
	trainWriter = csv.writer(trainFile)
	testWriter = csv.writer(testFile)
	trainWriter.writerow(["ID", "Class", "Class ID", "Object", "Transformation", "File Name", "Obj Start", "Obj End",
						  "Tr Start", "Tr End"])
	testWriter.writerow(["ID", "Class", "Class ID", "Object", "Transformation", "File Name", "Obj Start", "Obj End",
						  "Tr Start", "Tr End"])
	jpeg_frames_train = []
	jpeg_frames_test = []
	count = 0
	count_miss = 0
	obj_start_dict = {}
	obj_end_dict = {}
	tr_start_dict = {}
	tr_end_dict = {}
	rows_dict = {}
	images_dict = {}
	for fp in cropFilePaths:
		cropCSVFile = list(csv.DictReader(open(fp, "r")))
		for i in range(len(cropCSVFile)):
			row = cropCSVFile[i]
			cl = row['ca']
			obj = row['no']
			tr = row['tr']
			fr = row['fr']
			if (cl, obj) not in rows_dict.keys():
				rows_dict[(cl, obj)] = []
			if (cl, obj) not in images_dict.keys():
				images_dict[(cl, obj)] = []
			try:
				left, top, width, height = int(float(row['left'])), int(float(row['top'])), int(float(row['width'])), \
										   int(float(row['height']))
			except ValueError:
				count_miss += 1
				continue
			if cl in classesList and int(obj) <= numObjects:
				fileName = cl + "_" + obj.zfill(2) + "//" + cl + "_" + obj.zfill(2) + "_pivothead_" + tr + \
							 ".mp4_" + fr.zfill(3) + ".jpeg"
				imFilePath = imageDir + fileName
				try:
					assert(os.path.isfile(imFilePath))
				except AssertionError:
					count_miss += 1
					print(imFilePath)
				else:
					im = cv2.imread(filename = imFilePath)
					if cropImage:
						im_cropped = crop_image(imagePath = imFilePath, image = im, l = left, t = top, h = height, w = width)
					else:
						im_cropped = im
					im_resized = cv2.resize(im_cropped, (size, size), interpolation = cv2.INTER_CUBIC)
					count += 1
					_, im_encoded = cv2.imencode(".jpeg", im_resized)
					images_dict[(cl, obj)].append(im_encoded)
					rows_dict[(cl, obj)].append([cl, classesList.index(cl), obj, tr, fileName])
			if i % 200 == 0:
				print(i, time.time() - curr)
		print("Time taken for file", fp, ":", time.time() - curr)
		curr = time.time()
	row_count_train = 0
	row_count_test = 0
	for key in rows_dict.keys():
		for row_idx in range(len(rows_dict[key])):
			cl = key[0]
			obj = key[1]
			row = rows_dict[key][row_idx]
			tr = row[3]
			if int(obj) in TEST_NO[cl]:
				row_count = row_count_test
				row_count_test += 1
			else:
				row_count = row_count_train
				row_count_train += 1
			row = [row_count] + row
			rows_dict[key][row_idx] = row

			if (cl, obj) not in obj_start_dict.keys():
				obj_start_dict[(cl, obj)] = row_count
			obj_end_dict[(cl, obj)] = row_count

			if (cl, obj, tr) not in tr_start_dict.keys():
				tr_start_dict[(cl, obj, tr)] = row_count
			tr_end_dict[(cl, obj, tr)] = row_count

	for key in rows_dict.keys():
		for row_idx in range(len(rows_dict[key])):
			cl = key[0]
			obj = key[1]
			row = rows_dict[key][row_idx]
			tr = row[4]
			row = row + [obj_start_dict[(cl, obj)], obj_end_dict[(cl, obj)], tr_start_dict[(cl, obj, tr)],
						 tr_end_dict[(cl, obj, tr)]]
			rows_dict[key][row_idx] = row

	for key in rows_dict.keys():
		for row_idx in range(len(rows_dict[key])):
			cl = key[0]
			obj = key[1]
			row = rows_dict[key][row_idx]
			if int(obj) in TEST_NO[cl]:
				testWriter.writerow(row)
				jpeg_frames_test.append(images_dict[key][row_idx])
			else:
				trainWriter.writerow(row)
				jpeg_frames_train.append(images_dict[key][row_idx])

	print("Total Images Train:", len(jpeg_frames_train))
	print("Total Images Test:", len(jpeg_frames_test))
	print("Missed files:", count_miss)

	trainFile.close()
	testFile.close()
	f = open(trainPickle, 'wb')
	pickle.dump(jpeg_frames_train, f, pickle.HIGHEST_PROTOCOL)
	f.close()
	f = open(testPickle, 'wb')
	pickle.dump(jpeg_frames_test, f, pickle.HIGHEST_PROTOCOL)
	f.close()

	with open(trainImagesName, "rb") as f:
		encodedImages = pickle.load(f)
		print(len(encodedImages))
	with open(trainLabelsName, "r") as f:
		reader = csv.DictReader(f)
		print(len(list(reader)))

	with open(testImagesName, "rb") as f:
		encodedImages = pickle.load(f)
		print(len(encodedImages))
	with open(testLabelsName, "r") as f:
		reader = csv.DictReader(f)
		print(len(list(reader)))


# classes = ['airplane']
crop = True
numObjectsInClass = 30
toyboxFramesPath = "/home/sanyald/Documents/AIVAS/Projects/Toybox_frames/Toybox_New_Frame6_Size1920x1080/"
outNameCSV = "./data/toybox_data_interpolated"
outNamePickle = "./data/toybox_jpeg_frames_interpolated"
if crop:
	trainImagesName = outNameCSV + "_cropped_interpolated_train.pickle"
	trainLabelsName = outNameCSV + "_cropped_interpolated_train.csv"
	testImagesName = outNameCSV + "_cropped_interpolated_test.pickle"
	testLabelsName = outNameCSV + "_cropped_interpolated_test.csv"
	outNameCSV = outNameCSV + "_interpolated_cropped.csv"
	outNamePickle = outNamePickle + "_interpolated_cropped.pickle"
else:
	trainImagesName = outNameCSV + "_train.pickle"
	trainLabelsName = outNameCSV + "_train.csv"
	testImagesName = outNameCSV + "_test.pickle"
	testLabelsName = outNameCSV + "_test.csv"
	outNameCSV = outNameCSV + ".csv"
	outNamePickle = outNamePickle + ".pickle"

csvFiles = ["toybox_fps1_1080p_hodge_bbox.csv", "toybox_fps1_1080p_rot_bbox.csv"]
generate_data(classesList = classes, numObjects = numObjectsInClass, imageDir = toyboxFramesPath,
			  cropFilePaths = csvFiles, cropImage = crop, trainPickle = trainImagesName, trainCSV = trainLabelsName,
			  testPickle = testImagesName, testCSV = testLabelsName)
'''
x = 0
with open(trainImagesName, "rb") as f:
	imagesEncoded = pickle.load(f)
	start = 7633
	end = 7654
	for i in range(start, end):
		im = cv2.imdecode(imagesEncoded[i], 3)
		cv2.imshow(str(i), im)
		cv2.waitKey(0)
'''