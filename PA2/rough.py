import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math
import time
import random
import matlab.engine
import pdb
import open3d as o3d
from method import *

# im1 = cv2.imread("Data/sfm00.jpg")
# im2 = cv2.imread("Data/sfm01.jpg")
im3 = cv2.imread("Data/sfm03.jpg")
im4 = cv2.imread("Data/sfm04.jpg")

#### visualize ####
# cv2.imshow("im1", im1)
# cv2.waitKey(0)

keypoints, descriptors = cv2.SIFT_create().detectAndCompute(im3, None) 
keypoints2, descriptors2 = cv2.SIFT_create().detectAndCompute(im4, None)

# print(descriptors) # 2D array of 128-dim vectors
# print(descriptors.shape) # (64420,128)

#### visualize ####
# img_draw = cv2.drawKeypoints(im1, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img_draw =cv2.resize(img_draw, (1500,1000), fx=0.5, fy=0.5)
# cv2.imshow("im1", img_draw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors, descriptors2, k=2) # 37936 -> 3번 이미지에 맞춰짐 

good = []
good_examples = []
for i ,(m,n) in enumerate(matches):
    if m.distance < 0.9 * n.distance:
        good.append([m.queryIdx, m.trainIdx, m.distance]) #1429
        good_examples.append([m])

#### visualize ####
# matchesmask = [[0,0] for i in range(len(good))]
# draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesmask, flags = 0)
# res = cv2.drawMatchesKnn(im1, keypoints, im2, keypoints2, good_examples, None, **draw_params)
# plt.imshow(res)
# plt.show()
# cv2.waitKey(0)

matched_idx_1 = np.load("two_view_recon_info/sfm03_matched_idx.npy") # 2592
matched_idx_2 = np.load("two_view_recon_info/sfm04_matched_idx.npy") # 2592
descriptors_1 = np.load("two_view_recon_info/sfm03_descriptors.npy") # 37936,2, sift crate으로 구한 descriptor와 같음
descriptors_2 = np.load("two_view_recon_info/sfm04_descriptors.npy") # 36001, 2, sift crate으로 구한 descriptor와 같음
keypoints_1 = np.load("two_view_recon_info/sfm03_keypoints.npy") # 37936,2, BFMatcher로 구한 keypoints와 같음
keypoints_2 = np.load("two_view_recon_info/sfm04_keypoints.npy") # 36001, 2, BFMatcher로 구한 keypoints와 같음
inlinear = np.load("two_view_recon_info/inlinear.npy") # 1618
points_3d = np.load("two_view_recon_info/3D_points.npy") # 1618,3
p1 = np.load("two_view_recon_info/sfm03_camera_pose.npy")
p2 = np.load("two_view_recon_info/sfm04_camera_pose.npy")


matched_index = []
for i, j, _ in good: # good or matches
    if i in matched_idx_1 and j in matched_idx_2:
        matched_index.append(np.where(matched_idx_1 == i)[0][0])

inlinear_index = [np.where(inlinear==i) for i in matched_index]

cor = []
cor_2 = []
cor_3 = []
for i in inlinear_index:
    if i[0].size > 0: # empty list는 pass
      cor.append(points_3d[i]) # matched 된 3D point
      cor_2.append(keypoints_1[i]) # 3D point에 대응되는 2D point, 첫번째 이미지
      cor_3.append(keypoints_2[i]) # 3D point에 대응되는 2D point, 두번째 이미지 

idx = np.random.randint(0, len(cor), 3)
random_point = np.array(cor)[idx] # world coordinate, 3D
random_point_2 = np.array(cor_2)[idx] # image coordinate, 2D

file_path = "Data/intrinsic.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

data = []
for line in lines:
    line = line.strip('[];\n')  # [ ], ;, \n 문자를 제거
    row = [float(value) for value in line.split()]  # 공백으로 분할한 후 실수로 변환
    data.append(row)
df = pd.DataFrame(data)
k = df.to_numpy()
kinv = np.linalg.inv(k)

# nnc = np.ones(3)
# n_cor = []
# for i, p in enumerate(random_point):
#     nnc[:2] = keypoints_1[idx[i]] # keypoints가 맞나...?
#     temp = np.dot(kinv, nnc.T) #p norm
#     n_cor.append(temp.reshape(3))

# n_cor = np.array(n_cor)
# input_data = np.hstack((n_cor, random_point.reshape(3,3)))


# eng = matlab.engine.start_matlab()
# eng.addpath('functions/', nargout=0) 
# a = eng.PerspectiveThreePoint(input_data)
# eng.quit()

# camera_pose = np.array(a)
# print(camera_pose)
# print(camera_pose.shape)

# R = camera_pose[:3,:3]
# T = camera_pose[:3,3]

# C = - np.linalg.inv(R).dot(T) # camera center
# P = np.dot(R,random_point.reshape(3,3)) + np.array(T).reshape(3,1)


#### RanSac ####

threshold = 1  # Set your desired threshold

best_inliers = []
best_R = None
best_T = None

for _ in range(10):  # Repeat RANSAC for a certain number of iterations
    # Randomly select 3 keypoints and estimate camera pose
    idx = np.random.randint(0, len(cor), 3)
    random_point = np.array(cor)[idx] # world coordinate, 3D
    random_point_2 = np.array(cor_2)[idx] # image coordinate, 2D 
    sampled_points_2D = random_point_2
    sampled_points_3D = random_point.reshape(3, 3)

    # Estimate camera pose using the 3 selected keypoints

    nnc = np.ones(3)
    n_cor = []
    for i, p in enumerate(sampled_points_2D):
        nnc[:2] = cor_2[idx[i]] # keypoints가 맞나...?
        temp = np.dot(kinv, nnc.T) #p norm
        n_cor.append(temp.reshape(3))

    n_cor = np.array(n_cor)
    input_data = np.hstack((n_cor, random_point.reshape(3,3)))

    eng = matlab.engine.start_matlab()
    eng.addpath('functions/', nargout=0) 
    a = eng.PerspectiveThreePoint(input_data)
    eng.quit()

    camera_pose = np.array(a)

    try :
      R = camera_pose[:3,:3]
      T = camera_pose[:3,3]

      print(R)
      print(T)

      # C = - np.linalg.inv(R).dot(T) # camera center
      # P = np.dot(R,random_point.reshape(3,3)) + np.array(T).reshape(3,1)

      # Calculate reprojection error for all keypoints
      reproj_errors = []

      for i in range(len(cor_2)): # matched 된거만 조회? 
          # projected_point = np.dot(k, (np.dot(R, cor[i].T).T + T).T)
          # pdb.set_trace()
          world_to_camera = np.dot(R, cor[i].T).T + T 
          projected_point = np.dot(kinv,np.dot(k, p1).dot(np.append(world_to_camera,1)))
          projected_point2 = np.dot(kinv, np.dot(k, p2).dot(np.append(world_to_camera,1)))

          print("pro1" , projected_point)
          print("pro2" , projected_point2)
          # print(cor_2[i])
          print("\n")

          # pdb.set_trace()
          # reproj_error = np.linalg.norm(projected_point[:2] / projected_point[2] - cor_2[i].reshape(2,1))
          norm_1 = np.dot(kinv, np.append(cor_2[i],1).T)
          norm_2 = np.dot(kinv, np.append(cor_3[i],1).T)
          
          reproj_error_1 = np.linalg.norm(projected_point[:2] / projected_point[2] - norm_1[:2])
          reproj_error_2 = np.linalg.norm(projected_point2[:2] / projected_point2[2] - norm_2[:2])

          reproj_error = np.sqrt(reproj_error_1 + reproj_error_2)

          # print("reproj_error: ", reproj_error)
          reproj_errors.append(reproj_error)

          # pdb.set_trace()
      print("reproj_error: ", np.mean(reproj_error))

      # Count inliers based on the threshold
      inliers = [i for i in range(len(cor_2)) if reproj_errors[i] < threshold]

      if len(inliers) > len(best_inliers):
          # Update the best model
          best_inliers = inliers
          best_R = R
          best_T = T
    except:
        print("error!")
        pass
    
    print("best_inliers: ", len(best_inliers))
    print("best_R: ", best_R)
    print("best_T: ", best_T)


### triangulation ###
C = - np.linalg.inv(best_R).dot(best_T) # camera center
# P = np.dot(best_R,random_point.reshape(3,3)) + np.array(best_T).reshape(3,1)
P = np.hstack((best_R,best_T.reshape(3,1)))

x1 = np.ones(3)
x2 = np.ones(3)

A = np.zeros((4,4))

threeD = []
for i in range(len(cor_2)):
    x1[:2] = cor_2[i]
    x2[:2] = cor_3[i]

    norm_1 = np.dot(kinv, x1)
    norm_2 = np.dot(kinv, x2)

    # A1 = np.dot(np.dot(kinv,x1).reshape(1,3), p1)
    # A2 = np.dot(np.dot(kinv,x2).reshape(1,3), p2)
    # A = np.vstack((A1, A2))

    A[0,:] = norm_1[0]*p1[2,:] - p1[0,:]
    A[1,:] = norm_1[1]*p1[2,:] - p1[1,:]
    A[2,:] = norm_2[0]*P[2,:] - P[0,:]
    A[3,:] = norm_2[1]*P[2,:] - P[1,:]

    # pdb.set_trace()

    # print(A)   

    U, S, V = np.linalg.svd(A)

    X = V[-1,:3] / V[-1,-1]

    print(X)
    threeD.append(X)

threeD = np.array(threeD)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(threeD)
o3d.io.write_point_cloud("result/3D_points.ply", pcd)
# np.save("result/3D_points.npy", threeD)
# pdb.set_trace()