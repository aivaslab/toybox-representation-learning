import cv2
import pickle

trainFile = "./data/aloi_train.pickle"
pickleFile = pickle.load(open(trainFile, "rb"))
print(len(pickleFile))
k = 480
for i in range(48):
	img = cv2.imdecode(pickleFile[k + i], 3)
	cv2.imshow("djsdf", img)
	cv2.waitKey(0)
