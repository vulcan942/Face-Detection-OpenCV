import cv2
import numpy as np

img = cv2.imread('faces.jpg')


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.resize(img, (800,600))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray,1.1,5)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("faces Found",img)

cv2.waitKey(0)
