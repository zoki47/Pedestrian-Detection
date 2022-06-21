from turtle import width
import cv2,imutils
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
from imutils.video import FileVideoStream
import time
# Histogram of Oriented Gradients Detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
prev_frame_time = 0
new_frame_time = 0
cap = cv2.VideoCapture("Pedestrian_clip_3.mp4")

while True:
    ret,frame = cap.read()
    
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(frame,(5,5),0)

    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    pedestrians, weights = HOGCV.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.03)

    pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])

#primjena metode non-maxima suppression da se ukloni preklapanje okvira detektovanih pjesaka
    #pedestrians = non_max_suppression(pedestrians, probs=None, overlapThresh=0.5)
    
    count = 0
    # displej fpsa (frames per second)
    cv2.putText(frame, f'FPS :{fps}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#Oznacavanje pjesaka 
    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 20), (w,y), (0, 0, 255), -1)
        cv2.putText(frame, f'P{count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        count += 1
#Boja je jednaka prema broju pjesaka detektovanih na videu
    if count == 0:
        boja = 0,0,0 #crna
    elif count < 3:
        boja = 0,128,0 #zelena
    elif count > 3:
        boja = 0,255,255 #zuta
    pass
    cv2.putText(frame, f'Broj pjesaka :{count}', (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (boja), 2)
    print("Pjesak detektovan :", count,"\n""Fps :", fps)
    cv2.imshow('output', frame)

#Prekidanje videa tasterom Esc
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()


