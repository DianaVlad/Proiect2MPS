import cv2
import os
import filters as filterUtils
import sys
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) < 2:
	print('Usage: facedetection.py <image> -detect -f <filter1> <filter2> ...')
	exit(1)

faceDetection = False
if '-detect' in sys.argv:
	faceDetection = True

filters = []

if '-f' in sys.argv:
	for filter in sys.argv[sys.argv.index('-f') + 1 :]:
		filters.append(filter)

b=os.path.dirname(os.path.abspath(__file__))
os.chdir(b)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

img = cv2.imread(sys.argv[1])
imgNew = cv2.imread(sys.argv[1])

# Transforma img in alb-negru
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

imgNew = filterUtils.applyFilters(imgNew, filters)

if faceDetection == True:
	for (x,y,w,h) in faces:
		cv2.rectangle(imgNew,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = imgNew[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		smile = smile_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			# Detecta mai multi "ochi"
			# M-am gandit ca ochii pot fi doar de la jumatatea fetei in sus
			if(ey < h/2):
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		for (ex,ey,ew,eh) in smile:
			# Same, gura de la juamtatea fetei in jos
			if(ey > h/2):
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)

cv2.imshow('img',imgNew)
cv2.imwrite('test1.jpg', imgNew)
cv2.waitKey(0)
cv2.destroyAllWindows()