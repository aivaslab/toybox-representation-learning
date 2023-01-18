import pickle
import cv2

dataDirectory = "./data/"
fileName = "core50_imgs_ordered.pickle"
with open(dataDirectory + fileName, "rb") as ff:
	imgs = pickle.load(ff)

x = 1000
for i in range(x, x + 10):
	im = cv2.cvtColor(cv2.imdecode(imgs[i], 3), cv2.COLOR_RGB2BGR)
	cv2.imshow(str(i), im)
	cv2.waitKey(0)

