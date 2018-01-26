import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

region_of_interest_vertices = [
    (200,100),
    (-400, 400),
    (900, 450),
]
vid = cv2.VideoCapture('C:/Users/vinay/Desktop/video1.mp4', 0)
while(vid.isOpened()):
    ret,frame = vid.read()
    cropped_image = region_of_interest(
        frame,
        np.array([region_of_interest_vertices]),
    )
    cv2.imshow('cropped', cropped_image)
    cv2.imshow('orig', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
