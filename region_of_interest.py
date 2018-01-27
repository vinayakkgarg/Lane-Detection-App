import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


region_of_interest_vertices = [
    (175, 250),
    (-250, 400),
    (500, 400)
]
image = mpimg.imread('face.jpg')

cropped_image = region_of_interest(
    image,
    np.array([region_of_interest_vertices]),
)
plt.figure()
plt.imshow(cropped_image)
plt.show()
