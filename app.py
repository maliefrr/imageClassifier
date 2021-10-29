import cv2 as cv 

img1 = cv.imread('image/imageQuerry/Helios Trice Megistus.jpg',0)
img2 = cv.imread('image/imageTrain/4.jpg',0)

# load ORB
orb = cv.ORB_create(nfeatures=1000)

kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

# draw keypoint
imgKp1 = cv.drawKeypoints(img1,kp1,None)
imgKp2 = cv.drawKeypoints(img2,kp2,None)

# show keypoint
cv.imshow('Kp1',imgKp1)
cv.imshow('Kp2',imgKp2)

# matching descriptor
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good = []

for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

print(len(good))

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv.imshow('image 1',img1)
cv.imshow('image 2',img2)
cv.imshow('image 3',img3)
cv.waitKey(0)