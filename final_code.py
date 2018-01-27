import numpy as np
import cv2
import math

def region_of_interest(img, vertices):
    #Blank matrix that has image height/width.
    mask = np.zeros_like(img)
    mask_color = (255,255,255)
    #Fill in crop area
    cv2.fillPoly(mask, vertices, mask_color)
    #Returns the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

region_of_interest_vertices = [(200, 100),(-350, 400),(800, 400)]
#for video1 : (250, 25),(-300, 240),(800, 240)
#for video2 : (250, 25),(-300, 240),(800, 240)

def draw_lines(img, lines, color=[0, 0, 255], thickness=3):
    #No lines to draw, exit.
    if lines is None:
            exit(0)
    #Copy of original image.
    img = np.copy(img)
    #Blank image that matches the original image in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    #Draw lines on blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    #Merge blank image(with lines) with original image.
    img = cv2.addWeighted(img, 0.3, line_img, 1.0, 0.0)
    #Returns the modified image.
    return img

vid = cv2.VideoCapture('video1.mp4', 0)
thresholdval1 = 100
thresholdval2 = 200
minLineLength = 50
maxLineGap = 25

while(vid.isOpened()):
    ret,frame = vid.read()
    #Converts to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    #Applies Gaussian Blur
    gauss = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imshow('gauss+gray', gauss)
    #Apply Canny Edge Detector
    canny = cv2.Canny( gray, thresholdval1, thresholdval2)
    #Cropping the required region from canny image
    cropped = region_of_interest(
        canny,
        np.array([region_of_interest_vertices], np.int32),
    )
    cv2.imshow('cropped canny', cropped)
    #Cropping the required region from original video
    croppedOrig = region_of_interest(
        frame,
        np.array([region_of_interest_vertices], np.int32),
    )
    cv2.imshow('cropped original', croppedOrig)
    #Hough Space Transform applied to detect lines
    lines = cv2.HoughLinesP(
        cropped,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    #Prints coordinates of detected lines
    print(lines)
    line_img = draw_lines(frame,lines,thickness=5)
    cv2.imshow('Final', line_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
