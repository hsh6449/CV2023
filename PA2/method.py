import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import time
import random
import matlab.engine

def extract_and_match(img1, img2):
    # extract features and match
    # img1, img2: images
    # return: list of DMatch objects

    # extract features
    cv2.SIFT_create()

    # match features
    cv2.BFMatcher()
    
    NotImplementedError("extract_and_match() not implemented")

def rancsac(matches, kp1, kp2, img1, img2):
    # 3-point RANSAC
    # matches: list of DMatch objects
    # kp1, kp2: list of KeyPoint objects
    # img1, img2: images
    # return: list of inliers (DMatch objects)
    cv2.solveP3P()
    cv2.Rodrigues()
    
    NotImplementedError("3p_rancsac() not implemented")


def reconstruct(kp1, kp2, matches, img1, img2):
    # reconstruct 3D points (Traiangulation)
    # kp1, kp2: list of KeyPoint objects
    # matches: list of DMatch objects
    # img1, img2: images
    # return: list of 3D points (np.array)
    
    NotImplementedError("reconstruct() not implemented")


def plot_3d(points3D):
    # plot 3D points
    # points3D: list of 3D points (np.array)
    
    NotImplementedError("plot_3d() not implemented")

def BundleAdjustment():
    NotImplementedError("BundleAdjustment() not implemented")
