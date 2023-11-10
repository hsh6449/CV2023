import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import time
import random
import matlab.engine
import pdb

from utils import find_camera_pose

def extract_and_match(img1=None, img2=None, Three = True, Four = True):
    # extract features and match
    # img1, img2: images
    # return: list of DMatch objects

    # extract features
    if (Three == True) & (Four == True):
        kp1 = np.load("two_view_recon_info/sfm03_keypoints.npy")
        descriptors = np.load("two_view_recon_info/sfm03_descriptors.npy")

        kp2 = np.load("two_view_recon_info/sfm04_keypoints.npy")
        descriptors2 = np.load("two_view_recon_info/sfm04_descriptors.npy")

        matched_idx_1 = np.load("two_view_recon_info/sfm03_matched_idx.npy") # 2592
        matched_idx_2 = np.load("two_view_recon_info/sfm04_matched_idx.npy")

    elif (Four == True) & (Three == False):

        # 4번 이미지가 들어올때 무조건 4번이 2 -> 1에 대해 근사
        kp2 = np.load("two_view_recon_info/sfm04_keypoints.npy")
        descriptors2 = np.load("two_view_recon_info/sfm04_descriptors.npy")

        keypoints, descriptors = cv2.SIFT_create().detectAndCompute(img2, None)
        
        kp1 = []
        for i in range(len(keypoints)):
            kp1.append(keypoints[i].pt)
        kp1 = np.array(kp1)

        matched_idx_2 = np.load("two_view_recon_info/sfm04_matched_idx.npy")
    
    elif (Three == True) & (Four == False):
        # 3번 이미지가 들어올 때 무조건 3번이 2 -> 1번에 대해 근사
        kp2 = np.load("two_view_recon_info/sfm03_keypoints.npy")
        descriptors2 = np.load("two_view_recon_info/sfm03_descriptors.npy")

        keypoints, descriptors = cv2.SIFT_create().detectAndCompute(img2, None)

        kp1 = []
        for i in range(len(keypoints)):
            kp1.append(keypoints[i].pt)
        kp1 = np.array(kp1)

        matched_idx_2 = np.load("two_view_recon_info/sfm03_matched_idx.npy")

    else :
        # 무조건 1번에 대해서 근사할 것.
        keypoints, descriptors = cv2.SIFT_create().detectAndCompute(img1, None) 
        keypoints2, descriptors2 = cv2.SIFT_create().detectAndCompute(img2, None)

        kp1 = []
        kp2 = []
        for i in range(len(keypoints)):
            kp1.append(keypoints[i].pt)
            kp2.append(keypoints2[i].pt)
        kp1 = np.array(kp1)
        kp2 = np.array(kp2)




    # match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors2, k=2)

    # filtering
    good = []
    good_examples = []
    for i ,(m,n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            good.append([m.queryIdx, m.trainIdx, m.distance]) #1429
            good_examples.append([m])

    if (Three == False) or (Four == False):
        # Matching된 index 저장
        # 둘다 False면 둘 다 저장
        # 하나만 False면 1만 저장
        matched_idx_1 = np.array([i[0] for i in good]) # 3,4번 한개만 있을 때 부터 필요한거 
        matched_idx_2 = np.array([i[1] for i in good]) # 3, 4번 없을 때 필요한거 

        # pdb.set_trace()
    
    return good, kp1, kp2, matched_idx_1, matched_idx_2

def rancsac(matches, kp1, kp2, mc1, mc2, k, kinv, p1=None, p2=None, inlinear=None, points_3d = None):
    # 3-point RANSAC
    # matches: list of DMatch objects
    # kp1, kp2: list of KeyPoint objects
    # img1, img2: images
    # return: list of inliers (DMatch objects)

    matched_index = []

    for i, j, _ in matches:
        if i in mc1 and j in mc2: # filtering을 거쳤기 때문에 값이 없을 수도 있음
            matched_index.append(np.where(mc1 == i)[0][0])
    
    
    inlinear_index = [np.where(inlinear==i) for i in matched_index]
    # pdb.set_trace()
    

    ### inlinear 기반으로 matching된 점들을 cor에 저장
    ## cor : 3D point
    ## cor_2 : 2D point, 첫번째 이미지
    ## cor_3 : 2D point, 두번째 이미지

    cor = []
    cor_2 = []
    cor_3 = []
    inlinear_index_out = []
    for i in inlinear_index:
        if i[0].size > 0: # empty list는 pass
            inlinear_index_out.append(i[0][0])
            cor.append(points_3d[i]) # matched 된 3D point 
            cor_2.append(kp1[i]) # 3D point에 대응되는 2D point, 첫번째 이미지
            cor_3.append(kp2[i]) # 3D point에 대응되는 2D point, 두번째 이미지 

    threshold = 0.7  # Set threshold

    best_inliers = []
    best_R = None
    best_T = None

    for _ in range(10000):  # Repeat RANSAC for a certain number of iterations

        # Randomly select 3 keypoints and estimate camera pose
        idx = np.random.choice(range(0, len(cor)), 3, replace=False)
                
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
        input_data = np.hstack((n_cor, sampled_points_3D))

        camera_pose = find_camera_pose(input_data)

        ## camera pose가 값이 없을 수도 있기 때문에 값이 없는 경우에 다시 구하기
        try : 
            while (camera_pose.size == 0) : #or (camera_pose[:3,:4].shape[0] != 3) or (camera_pose[:3,:4].shape[1] != 4):

                idx = np.random.choice(range(0, len(cor)), 3, replace=False)
                    
                random_point = np.array(cor)[idx] # world coordinate, 3D
                random_point_2 = np.array(cor_2)[idx] # image coordinate, 2D 
                sampled_points_2D = random_point_2
                sampled_points_3D = random_point.reshape(3, 3)

                # Estimate camera pose using the 3 selected keypoints
                nnc = np.ones(3)
                n_cor = []
                for i, p in enumerate(sampled_points_2D):
                    nnc[:2] = cor_2[idx[i]] 
                    temp = np.dot(kinv, nnc.T) #p norm
                    n_cor.append(temp.reshape(3))

                n_cor = np.array(n_cor)
                input_data = np.hstack((n_cor, sampled_points_3D))
                camera_pose = find_camera_pose(input_data)

                if camera_pose.size != 0:
                    break
            
            ## 3,4번 이미지가 아닌경우에는 camerapose를 이용해서 p1을 구함
            if p1 is None:
                p1 = camera_pose[:3,:4]
        except:
            break

        try :
            R = camera_pose[:3,:3]
            T = camera_pose[:3,3]

            print(R)
            print(T)

            # Calculate reprojection error for all keypoints
            reproj_errors = []
            for i in range(len(cor_2)): # matched 된거만 조회? 
                world_to_camera = np.dot(R, cor[i].T).T + T 
                projected_point = np.dot(kinv,np.dot(k, p1).dot(np.append(world_to_camera,1)))
                projected_point2 = np.dot(kinv, np.dot(k, p2).dot(np.append(world_to_camera,1)))

                # print("pro1" , projected_point)
                # print("pro2" , projected_point2)
                # print("\n")

                norm_1 = np.dot(kinv, np.append(cor_2[i],1).T)
                norm_2 = np.dot(kinv, np.append(cor_3[i],1).T)
                
                reproj_error_1 = np.linalg.norm(projected_point[:2] / projected_point[2] - norm_1[:2])
                reproj_error_2 = np.linalg.norm(projected_point2[:2] / projected_point2[2] - norm_2[:2])

                reproj_error = np.sqrt(reproj_error_1 + reproj_error_2)
                reproj_errors.append(reproj_error)

            print("reproj_error: ", np.mean(reproj_error))

            # Count inliers based on the threshold
            inliers = [i for i in range(len(cor_2)) if reproj_errors[i] < threshold] # inlier는 인덱스, 개수만 세면 되므로  

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
    
    return best_R, best_T, p1, p2, cor, cor_2, cor_3, inlinear_index_out


def Traiangulation(matches, kp1, kp2, R, T, p1, p2, cor, cor_2, cor_3, k , kinv, inlinear_index, threeDinfo):
    # reconstruct 3D points (Traiangulation)
    # kp1, kp2: list of KeyPoint objects
    # matches: list of DMatch objects
    # img1, img2: images
    # return: list of 3D points (np.array)

    # C = - np.linalg.inv(R).dot(T) # camera center
    # P = np.hstack((R, T.reshape(3,1)))

    x1 = np.ones(3)
    x2 = np.ones(3)

    A = np.zeros((4,4))

    threeD = []
    reproj_errors = []
    # pdb.set_trace()
    for i in range(len(cor_2)):
        x1[:2] = cor_2[i]
        x2[:2] = cor_3[i]

        norm_1 = np.dot(kinv, x1)
        norm_2 = np.dot(kinv, x2)

        A[0,:] = norm_1[0]*p1[2,:] - p1[0,:]
        A[1,:] = norm_1[1]*p1[2,:] - p1[1,:]
        A[2,:] = norm_2[0]*p2[2,:] - p2[0,:]
        A[3,:] = norm_2[1]*p2[2,:] - p2[1,:]

        # pdb.set_trace()
        # print(A)   

        _, _, V = np.linalg.svd(A)

        X = V[-1,:3] / V[-1,-1] # 3D point, X
        # print(X)

        threeD.append(X)

        ## examination ##
        
        # reproj_error = np.sqrt(np.linalg.norm(X - cor[i])) # 이부분을 다시 생각해봐
        # reproj_errors.append(reproj_error)

        reproj_error = 0
        threshold = 10
        
        
        # pdb.set_trace()
        if int(inlinear_index[i]) in threeDinfo : # 이미 존재하는 3D point

            if R is None:
                pdb.set_trace()
            if T is None:   
                pdb.set_trace()

            proj2D = np.dot(R, cor[i].T).T + T # 레퍼런스 
            proj2D2 = np.dot(R, np.mean(threeDinfo[int(inlinear_index[i])]["3D"]).T).T + T # 추정값 

            # if TypeError:
            #     pdb.set_trace()
            # if ValueError:
            #     pdb.set_trace()
            # if IndexError:
            #     pdb.set_trace()
            # if np.isnan(proj2D).any():
            #     pdb.set_trace()
            # if np.isnan(proj2D2).any():
            #     pdb.set_trace()
            
            reproj_error = np.sqrt(np.linalg.norm(proj2D- proj2D2)) # 이미지로 투영시켜서 다시해야함
            # print(reproj_error)

            if threeDinfo[int(inlinear_index[i])]["valid"] == True: # valid한 3D point
                if reproj_error < threshold: # reproj_error가 0.75보다 작으면 
                    threeDinfo[int(inlinear_index[i])]["3D"].append(X)
                    threeDinfo[int(inlinear_index[i])]["valid"] = True
                else: # reproj_error가 0.75보다 크면
                    pass
            else:
                if reproj_error < threshold:
                    threeDinfo[int(inlinear_index[i])]["3D"].append(X)
                    threeDinfo[int(inlinear_index[i])]["valid"] = True
                else : 
                    pass

        else : # 존재하지 않는 3D point
            threeDinfo[int(inlinear_index[i])] =  {
                "3D" : [X], # 3D point가 여러개일 수 있으므로 list로 저장
                "valid" : False, # valid한 3D point인지 아닌지
            }

            # reproj_error = np.sqrt(np.linalg.norm(cor[i] - X)) # 이미지로 투영시켜서 다시해야함 + 3D point가 없으므로 cor[i]로 계산(이건 고민해봐야함)
            # print(reproj_error)

            # if reproj_error < threshold :
            #     threeDinfo[int(inlinear_index[i])]["3D"].append(X)
            #     threeDinfo[int(inlinear_index[i])]["valid"] = True
            # else:
            #     pass

    good_point = [threeDinfo[int(inlinear_index[i])]["3D"] for i in range(len(reproj_errors)) if reproj_errors[i] < threshold]
    # print("good poinrt : ", len(good_point))

    # threeD = np.array(threeD)
    
    return good_point, threeDinfo


def plot_3d(points3D):
    # plot 3D points
    # points3D: list of 3D points (np.array)
    
    NotImplementedError("plot_3d() not implemented")

def BundleAdjustment():
    NotImplementedError("BundleAdjustment() not implemented")
