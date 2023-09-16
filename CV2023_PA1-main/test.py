import numpy as np
import cv2
import matplotlib.pyplot as plt
from similarity import SAD

image_c = cv2.imread('input/04_noise25.png',0)
image_l = cv2.imread('input/03_noise25.png',0)
image_r = cv2.imread('input/05_noise25.png',0)


left_cost_volume, right_cost_volume, left_disparity, right_disparity = SAD(image_l, image_c, 24)

print(left_cost_volume.shape)