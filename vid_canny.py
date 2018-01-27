import numpy as np
import cv2

cap = cv2.VideoCapture('video1.mp4', 0)
thresholdval1 = 50
thresholdval2 = 100
while(cap.isOpened()):
    ret,frame = cap.read()
    frame = cv2.Canny( frame, thresholdval1, thresholdval2)
    frame_scaled = cv2.resize(frame,(500, 400))
    cv2.imshow('vid', frame_scaled)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
