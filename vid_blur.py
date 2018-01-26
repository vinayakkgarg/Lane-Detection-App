import numpy as np
import cv2

#Loading the video
vid = cv2.VideoCapture('Path for source video', 0)
while(vid.isOpened()):
    #Reading video frame by frame
    ret,frame = vid.read()
    #Applying the blur to the video
    gauss = cv2.GaussianBlur(frame, (15,15), 0)
    #Displaying the blurred video
    cv2.imshow('Gaussian Blur', gauss)
    #Displaying the original video
    cv2.imshow('original', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
