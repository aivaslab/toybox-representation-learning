import pickle
import numpy as np
import cv2
import csv
import math


testList = [3, 7, 10]
valList = [1, 5, 9]

dataDirectory = "./data/"
classes = ["plug adapters", "mobile phones", "scissors", "light bulbs", "cans", "glasses", "balls", "markers", "cups",
		   "remote controls"]

imagesFileName = dataDirectory + "core50_imgs.pickle"
labelsName = dataDirectory + "paths.pkl"
imagesOutNameDev = dataDirectory + "core50_data_dev_object.pickle"
imagesOutNameTrain = dataDirectory + "core50_data_train_object.pickle"
imagesOutNameTest = dataDirectory + "core50_data_test_object.pickle"
imagesOutNameVal = dataDirectory + "core50_data_val_object.pickle"

csvFileTrain = open(dataDirectory + "core50_data_train_object.csv", "w")
csvFileDev = open(dataDirectory + "core50_data_dev_object.csv", "w")
csvFileTest = open(dataDirectory + "core50_data_test_object.csv", "w")
csvFileVal = open(dataDirectory + "core50_data_val_object.csv", "w")

csvWriterTrain = csv.writer(csvFileTrain)
csvWriterDev = csv.writer(csvFileDev)
csvWriterTest = csv.writer(csvFileTest)
csvWriterVal = csv.writer(csvFileVal)

header = ["idx", "Class", "Class ID", "Object No", "Session No", "Frame No", "Sess Start", "Sess End", "Obj Start",
		  "Obj End", "Original Index"]
csvWriterTrain.writerow(header)
csvWriterDev.writerow(header)
csvWriterTest.writerow(header)
csvWriterVal.writerow(header)

images = pickle.load(open(imagesFileName, "rb"))
paths = pickle.load(open(labelsName, "rb"))
assert (len(paths) == len(images))

dict_id_train = {}
dict_frame_train = {}
images_list_train = []
dict_id_dev = {}
dict_frame_dev = {}
images_list_dev = []
dict_id_test = {}
dict_frame_test = {}
images_list_test = []
dict_id_val = {}
dict_frame_val = {}
images_list_val = []

for i in range(len(paths)):
	splits = paths[i].split("_")
	sessionNum = int(splits[1])
	objNum = int(splits[2])
	imID = int(splits[3].split(".")[0])
	cl_id = math.ceil(objNum/5.0) - 1
	# if sessionNum not in testList:
	if objNum % 5 > 0:
		if (cl_id, objNum, sessionNum) not in dict_id_train.keys():
			dict_id_train[(cl_id, objNum, sessionNum)] = []
			dict_frame_train[(cl_id, objNum, sessionNum)] = []
		dict_id_train[(cl_id, objNum, sessionNum)].append(i)
		dict_frame_train[(cl_id, objNum, sessionNum)].append(imID)
		# if sessionNum not in valList:
		if objNum % 5 < 4:
			if (cl_id, objNum, sessionNum) not in dict_id_dev.keys():
				dict_id_dev[(cl_id, objNum, sessionNum)] = []
				dict_frame_dev[(cl_id, objNum, sessionNum)] = []
			dict_id_dev[(cl_id, objNum, sessionNum)].append(i)
			dict_frame_dev[(cl_id, objNum, sessionNum)].append(imID)
		else:
			if (cl_id, objNum, sessionNum) not in dict_id_val.keys():
				dict_id_val[(cl_id, objNum, sessionNum)] = []
				dict_frame_val[(cl_id, objNum, sessionNum)] = []
			dict_id_val[(cl_id, objNum, sessionNum)].append(i)
			dict_frame_val[(cl_id, objNum, sessionNum)].append(imID)
	else:
		if (cl_id, objNum, sessionNum) not in dict_id_test.keys():
			dict_id_test[(cl_id, objNum, sessionNum)] = []
			dict_frame_test[(cl_id, objNum, sessionNum)] = []
		dict_id_test[(cl_id, objNum, sessionNum)].append(i)
		dict_frame_test[(cl_id, objNum, sessionNum)].append(imID)

count_train = 0
count_dev = 0
count_test = 0
count_val = 0

obj_start_dict_train = {}
obj_end_dict_train = {}
sess_start_dict_train = {}
sess_end_dict_train = {}

obj_start_dict_dev = {}
obj_end_dict_dev = {}
sess_start_dict_dev = {}
sess_end_dict_dev = {}

obj_start_dict_test = {}
obj_end_dict_test = {}
sess_start_dict_test = {}
sess_end_dict_test = {}

obj_start_dict_val = {}
obj_end_dict_val = {}
sess_start_dict_val = {}
sess_end_dict_val = {}

for cl_id in range(10):
	objs = [5 * cl_id + 1, 5 * cl_id + 2, 5 * cl_id + 3, 5 * cl_id + 4, 5 * cl_id + 5]
	for obj in objs:
		for sess in range(1, 12):
			# if sess not in testList:
			if obj % 5 > 0:
				idArr_train = dict_id_train[(cl_id, obj, sess)]
				frameArr_train = dict_frame_train[(cl_id, obj, sess)]
				assert (len(idArr_train) == len(frameArr_train))
				for idx in range(len(idArr_train)):
					if obj not in obj_start_dict_train.keys():
						obj_start_dict_train[obj] = count_train
						obj_end_dict_train[obj] = count_train
					elif count_train > obj_end_dict_train[obj]:
						obj_end_dict_train[obj] = count_train
					if (obj, sess) not in sess_start_dict_train.keys():
						sess_start_dict_train[(obj, sess)] = count_train
						sess_end_dict_train[(obj, sess)] = count_train
					elif count_train > sess_end_dict_train[(obj, sess)]:
						sess_end_dict_train[(obj, sess)] = count_train

					count_train += 1

				# if sess not in valList:
				if obj % 5 < 4:
					idArr_dev = dict_id_dev[(cl_id, obj, sess)]
					frameArr_dev = dict_frame_dev[(cl_id, obj, sess)]
					assert (len(idArr_dev) == len(frameArr_dev))
					for idx in range(len(idArr_dev)):
						if obj not in obj_start_dict_dev.keys():
							obj_start_dict_dev[obj] = count_dev
							obj_end_dict_dev[obj] = count_dev
						elif count_dev > obj_end_dict_dev[obj]:
							obj_end_dict_dev[obj] = count_dev

						if (obj, sess) not in sess_start_dict_dev.keys():
							sess_start_dict_dev[(obj, sess)] = count_dev
							sess_end_dict_dev[(obj, sess)] = count_dev
						elif count_dev > sess_end_dict_dev[(obj, sess)]:
							sess_end_dict_dev[(obj, sess)] = count_dev
						count_dev += 1
				else:
					idArr_val = dict_id_val[(cl_id, obj, sess)]
					frameArr_val = dict_frame_val[(cl_id, obj, sess)]
					assert (len(idArr_val) == len(frameArr_val))
					for idx in range(len(idArr_val)):
						if obj not in obj_start_dict_val.keys():
							obj_start_dict_val[obj] = count_val
							obj_end_dict_val[obj] = count_val
						elif count_val > obj_end_dict_val[obj]:
							obj_end_dict_val[obj] = count_val
						if (obj, sess) not in sess_start_dict_val.keys():
							sess_start_dict_val[(obj, sess)] = count_val
							sess_end_dict_val[(obj, sess)] = count_val
						elif count_val > sess_end_dict_val[(obj, sess)]:
							sess_end_dict_val[(obj, sess)] = count_val

						count_val += 1
			else:
				idArr_test = dict_id_test[(cl_id, obj, sess)]
				frameArr_test = dict_frame_test[(cl_id, obj, sess)]
				assert (len(idArr_test) == len(frameArr_test))
				for idx in range(len(idArr_test)):
					if obj not in obj_start_dict_test.keys():
						obj_start_dict_test[obj] = count_test
						obj_end_dict_test[obj] = count_test
					elif count_test > obj_end_dict_test[obj]:
						obj_end_dict_test[obj] = count_test
					if (obj, sess) not in sess_start_dict_test.keys():
						sess_start_dict_test[(obj, sess)] = count_test
						sess_end_dict_test[(obj, sess)] = count_test
					elif count_test > sess_end_dict_test[(obj, sess)]:
						sess_end_dict_test[(obj, sess)] = count_test

					count_test += 1

count_train = 0
count_dev = 0
count_test = 0
count_val = 0
for cl_id in range(10):
	objs = [5 * cl_id + 1, 5 * cl_id + 2, 5 * cl_id + 3, 5 * cl_id + 4, 5 * cl_id + 5]
	for obj in objs:
		for sess in range(1, 12):
			# if sess not in testList:
			if obj % 5 > 0:
				idArr_train = dict_id_train[(cl_id, obj, sess)]
				frameArr_train = dict_frame_train[(cl_id, obj, sess)]
				assert (len(idArr_train) == len(frameArr_train))
				for idx in range(len(idArr_train)):
					csvWriterTrain.writerow([count_train, classes[cl_id], cl_id, obj, sess, frameArr_train[idx],
											 sess_start_dict_train[(obj, sess)], sess_end_dict_train[(obj, sess)],
											 obj_start_dict_train[obj], obj_end_dict_train[obj], idArr_train[idx]])
					count_train += 1
					im = images[idArr_train[idx]]
					im_d = cv2.imdecode(im, 3)
					im_r = cv2.resize(im_d, (224, 224))
					_, im = cv2.imencode(".jpeg", im_r)
					images_list_train.append(im)

				# if sess not in valList:
				if obj % 5 < 4:
					idArr_dev = dict_id_dev[(cl_id, obj, sess)]
					frameArr_dev = dict_frame_dev[(cl_id, obj, sess)]
					assert (len(idArr_dev) == len(frameArr_dev))
					for idx in range(len(idArr_dev)):
						csvWriterDev.writerow(
							[count_dev, classes[cl_id], cl_id, obj, sess, frameArr_dev[idx],
							 sess_start_dict_dev[(obj, sess)], sess_end_dict_dev[(obj, sess)],
							 obj_start_dict_dev[obj], obj_end_dict_dev[obj], idArr_dev[idx]])
						count_dev += 1
						im = images[idArr_dev[idx]]
						im_d = cv2.imdecode(im, 3)
						im_r = cv2.resize(im_d, (224, 224))
						_, im = cv2.imencode(".jpeg", im_r)
						images_list_dev.append(im)
				else:
					idArr_val = dict_id_val[(cl_id, obj, sess)]
					frameArr_val = dict_frame_val[(cl_id, obj, sess)]
					assert (len(idArr_val) == len(frameArr_val))
					for idx in range(len(idArr_val)):
						csvWriterVal.writerow(
							[count_val, classes[cl_id], cl_id, obj, sess, frameArr_val[idx],
							 sess_start_dict_val[(obj, sess)], sess_end_dict_val[(obj, sess)],
							 obj_start_dict_val[obj], obj_end_dict_val[obj], idArr_val[idx]])
						count_val += 1
						im = images[idArr_val[idx]]
						im_d = cv2.imdecode(im, 3)
						im_r = cv2.resize(im_d, (224, 224))
						_, im = cv2.imencode(".jpeg", im_r)
						images_list_val.append(im)
			else:
				idArr_test = dict_id_test[(cl_id, obj, sess)]
				frameArr_test = dict_frame_test[(cl_id, obj, sess)]
				assert (len(idArr_test) == len(frameArr_test))
				for idx in range(len(idArr_test)):
					csvWriterTest.writerow([count_test, classes[cl_id], cl_id, obj, sess, frameArr_test[idx],
											sess_start_dict_test[(obj, sess)], sess_end_dict_test[(obj, sess)],
											obj_start_dict_test[obj], obj_end_dict_test[obj], idArr_test[idx]])
					count_test += 1
					im = images[idArr_test[idx]]
					im_d = cv2.imdecode(im, 3)
					im_r = cv2.resize(im_d, (224, 224))
					_, im = cv2.imencode(".jpeg", im_r)
					images_list_test.append(im)

with open(imagesOutNameTrain, "wb") as outImagesFileTrain:
	pickle.dump(images_list_train, outImagesFileTrain, pickle.DEFAULT_PROTOCOL)
csvFileTrain.close()
outImagesFileTrain.close()
with open(imagesOutNameDev, "wb") as outImagesFileDev:
	pickle.dump(images_list_dev, outImagesFileDev, pickle.DEFAULT_PROTOCOL)
csvFileDev.close()
outImagesFileDev.close()
with open(imagesOutNameTest, "wb") as outImagesFileTest:
	pickle.dump(images_list_test, outImagesFileTest, pickle.DEFAULT_PROTOCOL)
csvFileTest.close()
outImagesFileTest.close()
with open(imagesOutNameVal, "wb") as outImagesFileVal:
	pickle.dump(images_list_val, outImagesFileVal, pickle.DEFAULT_PROTOCOL)
csvFileVal.close()
outImagesFileVal.close()
print(count_dev, count_val, count_train, count_test)
assert(count_dev + count_val == count_train)
assert(count_train + count_test == len(paths))
# print(obj_start_dict_train)
# print(obj_end_dict_train)
