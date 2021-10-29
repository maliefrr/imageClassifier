# load semua library
import cv2 as cv 
import numpy as np
import os

# keypoint and descriptor extraction
orb = cv.ORB_create(nfeatures=1000)

# import images
path = 'image/imageQuerry'
images = []
classNames = []
myList = os.listdir(path)
print('Total Class Detected = ',len(myList))

# split nama image yang di load lalu meletakkannya kedalam array
for cl in myList:
    imgCur = cv.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(cl.split('.jpg')[0])

# mengekstrak deskriptor
def findDes(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

# mengekstrak id dari gambar
def findID(img,desList,thres=15):
    kp2,des2 = orb.detectAndCompute(img,None)

    # matching descriptor
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des,des2,k=2)
            good = []
            for m , n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    # print(matchList)
    if len(matchList) != 0:
        if(max(matchList) > thres):
            finalVal = matchList.index(max(matchList))
    return finalVal
    
    


desList = findDes(images)
print(len(desList))



cap = cv.VideoCapture(0)

# infinite loop untuk menjalankan video capture
while True:


    success,img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    id = findID(img2,desList)

    if id != -1:
        cv.putText(imgOriginal,classNames[id],(50,50),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

    cv.imshow('image 2',imgOriginal)

    cv.waitKey(1)
