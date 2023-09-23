import numpy as np
import cv2
import matplotlib.pyplot as plt
from similarity import SAD, SSD
from aggregate_cost_volume import generate_cor_pass, aggregate_cost_volume

image_c = cv2.imread('input/04_noise25.png', 0)
image_l = cv2.imread('input/03_noise25.png', 0)
image_r = cv2.imread('input/05_noise25.png', 0)


left_cost_volume, right_cost_volume, left_disparity, right_disparity = SAD(
    image_l, image_c, 24)

left_cost_volume2, right_cost_volume2, left_disparity2, right_disparity2 = SSD(
    image_l, image_c, 24)

# print(image_l.shape)
# print(np.min(left_cost_volume2, axis=0))
# print(left_disparity2.shape)

# agg_temp = np.min(left_cost_volume, axis=0)

# sample_disparity = cv2.imread(left_disparity, 0)
# cv2.imshow("disparity", (right_disparity*10).astype(np.uint8))
# cv2.imshow("disparity", (left_disparity2*10).astype(np.uint8))
# cv2.waitKey(0)

# test code
forward_pass = generate_cor_pass(left_cost_volume, 1)
a = aggregate_cost_volume(left_cost_volume)
print(a)

# print(-np.zeros((24, 215, 328)))

# for i in range(24):
#     print(i)
