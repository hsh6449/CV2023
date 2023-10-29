import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import time
import random
import glob
import matlab.engine

from method import extract_and_match, rancsac, reconstruct, plot_3d


def main():
    
    images = glob.glob('Data/*.png')

    # load images
    # 대충 느낌만 만들어 놓은 거라서 다시 짜야 함  
    for i in range(len(images)):
        img1 = cv2.imread(images[i])
        img1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        img2 = cv2.imread(images[i+1])
        img2 = cv2.cvtColor(images[i+1], cv2.COLOR_BGR2GRAY)

        # extract features and match
        matches, kp1, kp2 = extract_and_match(img1, img2)

        # 3-point RANSAC
        inliers = rancsac(matches, kp1, kp2, img1, img2)

        # reconstruct 3D points
        points3D = reconstruct(kp1, kp2, inliers, img1, img2)

        # plot 3D points
        plot_3d(points3D)


if __name__ == '__main__':
    main()