import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import time
import random
import matlab.engine

from method import *

im1 = cv2.imread("Data/sfm00.jpg")
im2 = cv2.imread("Data/sfm01.jpg")

# cv2.imshow("im1", im1)
# cv2.waitKey(0)

keypoints, descriptors = cv2.SIFT_create().detectAndCompute(im1, None)
keypoints2, descriptors2 = cv2.SIFT_create().detectAndCompute(im2, None)
print(descriptors) # 2D array of 128-dim vectors
print(descriptors.shape) # (64420,128)


# img_draw = cv2.drawKeypoints(im1, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img_draw =cv2.resize(img_draw, (1500,1000), fx=0.5, fy=0.5)

# cv2.imshow("im1", img_draw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

matches = cv2.BFMatcher().knnMatch(descriptors, descriptors2, k=2)
# print(matches)
# sorted_matches = sorted(matches, key = lambda x : x.distance)
res = cv2.drawMatchesKnn(im1, keypoints, im2, keypoints2, matches, None, flags = 2).resize((1500,1000), fx=0.5, fy=0.5)

 
cv2.imshow('res', res)
cv2.waitKey(0)