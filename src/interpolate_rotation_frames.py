import csv
import cv2
import os

fileName = "./toybox_rot_frames_fixed.csv"
pathToVids = "/home/sanyald/Documents/AIVAS/Projects/Toybox_frames/Toybox_New_Frame6_Size1920x1080/"
outFile = "./toybox_rot_frames_interpolated.csv"
outWriter = csv.writer(open(outFile, "w"))
outWriter.writerow(["left", "top", "width", "height", "ca", "no", "tr", "fr"])
readFile = open(fileName, "r")
rows = list(csv.DictReader(readFile))
print(len(rows))
row = rows[0]
countEnds = 0
countErr = 0
frames = []
save = False
for i in range(len(rows) - 1):
	cl = row['ca']
	obj = int(row['no'])
	tr = row['tr']
	fr = int(row['fr'])
	imgPath = pathToVids + cl + "_" + str(obj).zfill(2) + "/" + cl + "_" + str(obj).zfill(2) + \
			  "_pivothead_" + tr + ".mp4_" + str(fr).zfill(3) + ".jpeg"
	assert os.path.isfile(imgPath)
	# img = cv2.imread(imgPath)
	assert(tr == "rxminus" or tr == "rxplus" or tr == "ryminus" or tr == "ryplus" or tr == "rzplus" or tr == "rzminus")
	nextRow = rows[i + 1]
	nextCl = nextRow['ca']
	nextObj = int(nextRow['no'])
	nextTr = nextRow['tr']
	nextFr = int(nextRow['fr'])
	assert(nextTr == "rxminus" or nextTr == "rxplus" or nextTr == "ryminus" or nextTr == "ryplus" or nextTr == "rzplus"
		   or nextTr == "rzminus")
	if cl == nextCl and obj == nextObj and tr == nextTr:
		l, t, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])

		prevL, prevT, prevW, prevH = rows[i - 1]['left'], rows[i - 1]['top'], rows[i - 1]['width'], rows[i - 1][
			'height']
		nextL, nextT, nextW, nextH = rows[i + 1]['left'], rows[i + 1]['top'], rows[i + 1]['width'], rows[i + 1][
			'height']
		outWriter.writerow([l, t, w, h, cl, obj, tr, fr])
		if i > 0:
			sizePrev = int(prevW) * int(prevH)
			sizeNext = int(nextW) * int(nextH)
			if sizePrev < sizeNext:
				sizeLess = sizePrev
			else:
				sizeLess = sizeNext
			if w*h < 0.3 * sizeLess:
				print(i, cl, obj, tr, fr, h, w)
				l = int((int(prevL) + int(nextL))/2)
				t = int((int(prevT) + int(nextT))/2)
				h = int((int(prevH) + int(nextH))/2)
				w = int((int(prevW) + int(nextW))/2)
				if not save:
					countErr += 1
					save = True
		# cv2.rectangle(img, (l, t), (l + w, t + h), (0, 0, 0), 5)
		# frames.append(img)
		for intFrame in range(fr + 2, nextFr, 2):
			intL = int(l + ((int(nextL) - l)*(intFrame - fr))/(nextFr - fr))
			intT = int(t + ((int(nextT) - t)*(intFrame - fr))/(nextFr - fr))
			intW = int(w + ((int(nextW) - w)*(intFrame - fr))/(nextFr - fr))
			intH = int(h + ((int(nextH) - h)*(intFrame - fr))/(nextFr - fr))
			outWriter.writerow([intL, intT, intW, intH, cl, obj, tr, intFrame])
	else:
		countEnds += 1

		if save:
			fileName = "../output" + cl + "_" + str(obj) + "_" + tr + ".avi"
			print("Writing video to", fileName)
			# out = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (1920, 1080))
			# for img in frames:
				# out.write(img)
			# out.release()
		frames = []
		save = False
	row = nextRow

print(countEnds)
print(countErr)
