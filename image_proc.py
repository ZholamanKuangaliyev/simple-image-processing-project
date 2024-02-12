import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

### -------------------------------< Code >------------------------------- ###

scale = 1
radius = 5
diameter = radius * 2
num_of_colors = 10
div = 32
file_name = "image1.jpg"

org_image = cv2.imread(file_name)
org_image = cv2.resize(org_image, ((org_image.shape[1] // scale) // diameter * diameter, (org_image.shape[0] // scale) // diameter * diameter), interpolation = cv2.INTER_AREA)
WIDTH, HEIGHT = org_image.shape[:2]
img = Image.open(file_name)

### Finding dominant colors of the original image
kmeans = KMeans(n_clusters = num_of_colors, random_state = 42).fit(org_image.reshape((-1, 3)))
labels = kmeans.labels_
centers = kmeans.cluster_centers_
img_less_colors = centers[labels].reshape(org_image.shape).astype('uint8')
img_less_colors = cv2.resize(img_less_colors, (img_less_colors.shape[1] // diameter, img_less_colors.shape[0] // diameter), interpolation = cv2.INTER_AREA)

### Finding BG color for image
bg_color = list(sorted(img.getcolors(2 ** 24), reverse = True)[0][1])

### Making image with the same size filled with the BG color
image = np.zeros((WIDTH, HEIGHT, 3), dtype = np.uint8)
image[:] = bg_color

for y in range(img_less_colors.shape[0]):
    for x in range(img_less_colors.shape[1]):
        color = img_less_colors[y][x]
        color = ( int (color[0]), int (color[1]), int (color[2]))
        if (math.dist(color, bg_color) > 10):
            image = cv2.circle(image, (x * diameter + radius, y * diameter + radius), radius - 1, tuple (color), -1)
        
image = cv2.convertScaleAbs(image, alpha = 1.1, beta = 3)

cv2.imshow("Image", org_image)
cv2.imshow("processed image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()