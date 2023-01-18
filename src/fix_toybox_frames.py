import os

dirPath = "/home/sanyald/Documents/AIVAS/Projects/Toybox_frames/Toybox_New_Frame6_Size1920x1080/"


# Fix Cup 21 all files from truck
cl = "cup"
obj = 21
objDirPath = dirPath + cl + "_" + str(obj).zfill(2) + "/"
filesInDir = os.listdir(objDirPath)
print(len(filesInDir))
count = 0
for file in filesInDir:
	filePath = objDirPath + file
	if filePath.find("truck") != -1:
		count += 1
		changedFileName = file.replace("truck", "cup")
		changedFilePath = objDirPath + changedFileName
		print(filePath, changedFilePath)
		os.rename(filePath, changedFilePath)
print(count)

# Fix car 3 rzminus from rzminusb

cl = "car"
obj = 3
tr = "rzminusss."
tr_actual = "rzminus."
objDirPath = dirPath + cl + "_" + str(obj).zfill(2) + "/"
filesInDir = os.listdir(objDirPath)
print(len(filesInDir))
count = 0
for file in filesInDir:
	filePath = objDirPath + file
	if filePath.find(tr) != -1:
		count += 1
		changedFileName = file.replace(tr, tr_actual)
		changedFilePath = objDirPath + changedFileName
		print(filePath, changedFilePath)
		os.rename(filePath, changedFilePath)
print(count)

# Fix duck 6 rxminus from rxminu
cl = "duck"
obj = 6
tr = "rxminusssss."
tr_actual = "rxminus."
objDirPath = dirPath + cl + "_" + str(obj).zfill(2) + "/"
filesInDir = os.listdir(objDirPath)
print(len(filesInDir))
count = 0
for file in filesInDir:
	filePath = objDirPath + file
	if filePath.find(tr) != -1:
		count += 1
		changedFileName = file.replace(tr, tr_actual)
		changedFilePath = objDirPath + changedFileName
		print(filePath, changedFilePath)
		os.rename(filePath, changedFilePath)
print(count)

# Fix giraffe 2 rzplus from rplus
cl = "giraffe"
obj = 2
tr = "rplus."
tr_actual = "rzplus."
objDirPath = dirPath + cl + "_" + str(obj).zfill(2) + "/"
filesInDir = os.listdir(objDirPath)
print(len(filesInDir))
count = 0
for file in filesInDir:
	filePath = objDirPath + file
	if filePath.find(tr) != -1:
		count += 1
		changedFileName = file.replace(tr, tr_actual)
		changedFilePath = objDirPath + changedFileName
		print(filePath, changedFilePath)
		os.rename(filePath, changedFilePath)
print(count)

