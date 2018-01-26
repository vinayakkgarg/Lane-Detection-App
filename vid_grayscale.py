import numpy as np
import cv2

#Loading the video
vid = cv2.VideoCapture('Path for source video', 0)
while(vid.isOpened()):
    #Reading video frame by frame
    ret,frame = vid.read()
    #Converting video to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Displaying the grayscaled video
    cv2.imshow('gray', gray)
    #Displaying the original video
    cv2.imshow('original', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
