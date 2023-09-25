import numpy as np
import cv2
import matplotlib.pyplot as plt
from similarity import SAD, SSD
from aggregate_cost_volume import generate_cor_pass, aggregate_cost_volume
from warp import warp_image
from Semi_Global_Matching import semi_global_matching

# -1은 원래대로 읽어오는 것, 0은 흑백으로 읽어오는 것
image_c = cv2.imread('input/04_noise25.png', 0)
image_l = cv2.imread('input/01_noise25.png', 0)
image_r = cv2.imread('input/07_noise25.png', 0)

image_lm = cv2.imread('input/01_noise25.png', -1)


left_cost_volume, right_cost_volume, left_disparity, right_disparity = SAD(
    image_l, image_c, 24)

# left_cost_volume2, right_cost_volume2, left_disparity2, right_disparity2 = SSD(
#     image_l, image_c, 24)

# print(image_l.shape)
# print(np.min(left_cost_volume2, axis=0))
# print(left_disparity2.shape)

# agg_temp = np.min(left_cost_volume, axis=0)

# sample_disparity = cv2.imread(left_disparity, 0)
# cv2.imshow("disparity", (right_disparity*10).astype(np.uint8))
# cv2.imshow("disparity", (left_disparity2*10).astype(np.uint8))
# cv2.waitKey(0)

# test code
forward_pass = generate_cor_pass(left_cost_volume, size=1)
a = semi_global_matching(image_l, image_r, 24)
print("aggregated_disparity : ", a.shape)

warped_image = warp_image(image_lm, a, loc='left')
print(warped_image)

# print(-np.zeros((24, 215, 328)))

# for i in range(24):
#     print(i)
# modified = cv2.cvtColor(image_c, cv2.COLOR_RGB2GRAY)  # (215, 328) 회색 이미지는 2차원
# modified2 = cv2.cvtColor(modified, cv2.COLOR_GRAY2RGB)

# print(modified2.shape)
cv2.imshow("test", (warped_image).astype(np.uint8))
cv2.waitKey(0)
