import numpy as np
import cv2

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    # Returning the image only where mask pixels matc
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

region_of_interest_vertices = [
    (200,100),
    (-400, 400),
    (900, 450),
]

#Loading the video
vid = cv2.VideoCapture('Path for source video', 0)
while(vid.isOpened()):
    #Reading the video frame by frame
    ret,frame = vid.read()
    #Cropping the video to remove unnecessary parts
    cropped_image = region_of_interest(
        frame,
        np.array([region_of_interest_vertices]),
    )
    #Display the cropped video file
    cv2.imshow('cropped', cropped_image)
    #Display the original video file
    cv2.imshow('orig', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
