import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math
import time
import random
import glob
import matlab.engine
import open3d as o3d

from method import extract_and_match, rancsac, Traiangulation, plot_3d


def main():
    
    images = glob.glob('Data/*.jpg')

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

    iter = 0

    inlinear34 = np.load("two_view_recon_info/inlinear.npy")
    points_3d34 = np.load("two_view_recon_info/3D_points.npy")

    threeDinof = {}
    threedpoints = []

    while iter < 50:
        iter += 1
        print("iter : ", iter)

        if iter == 1 :
            # 첫번째는 무조건 3, 4번 이미지
            img1 = cv2.imread(images[3])
            img2 = cv2.imread(images[4])

            matches, kp1, kp2, mc1, mc2 = extract_and_match(img1, img2, Three = True, Four = True)
            inlinear = np.load("two_view_recon_info/inlinear.npy") 
            points_3d = np.load("two_view_recon_info/3D_points.npy")

            p1 = np.load("two_view_recon_info/sfm03_camera_pose.npy")
            p2 = np.load("two_view_recon_info/sfm04_camera_pose.npy")
        
        elif 1 < iter < 5:
            # 2~4im번은 3번과 0,1,2 번 이미지
            img1 = cv2.imread(images[3])
            img2 = cv2.imread(images[iter - 2])

            matches, kp1, kp2, mc1, mc2 = extract_and_match(img1, img2, Three = True, Four = False)
            np.save(f"two_view_recon_info/sfm{iter - 2}_matched_idx.npy", mc1)

            inlinear = np.load("two_view_recon_info/inlinear.npy") # 3번 이미지 기준으로 R,t를 근사할 것이기 때문에 사용가능 
            points_3d = np.load("two_view_recon_info/3D_points.npy")

            p1 = None
            p2 = np.load("two_view_recon_info/sfm03_camera_pose.npy")

        elif 4 < iter < 15:
            # 5~14 번은 3번과 5~14번 이미지
            img1 = cv2.imread(images[3])
            img2 = cv2.imread(images[iter])

            matches, kp1, kp2, mc1, mc2 = extract_and_match(img1, img2, Three = True, Four = False)
            np.save(f"two_view_recon_info/sfm{iter}_matched_idx.npy", mc1)

            inlinear = np.load("two_view_recon_info/inlinear.npy") # 3번 이미지 기준으로 R,t를 근사할 것이기 때문에 사용가능
            points_3d = np.load("two_view_recon_info/3D_points.npy")

            p1 = None
            p2 = np.load("two_view_recon_info/sfm03_camera_pose.npy")
            
        elif 14 < iter < 25 :
            # 15~24번은 4번과 5~14번 이미지
            img1 = cv2.imread(images[4])
            img2 = cv2.imread(images[iter-10])

            matches, kp1, kp2, mc1, mc2  = extract_and_match(img1, img2, Three = False, Four = True)
            np.save(f"two_view_recon_info/sfm{iter-10}_matched_idx.npy", mc1)

            inlinear = np.load("two_view_recon_info/inlinear.npy") # 4번 이미지 기준으로 R,t를 근사할 것이기 때문에 사용가능
            points_3d = np.load("two_view_recon_info/3D_points.npy")

            p1 = None
            p2 = np.load("two_view_recon_info/sfm04_camera_pose.npy")

        elif 24 < iter < 28 :
            # 25~27번은 4번과 0,1,2번 이미지
            img1 = cv2.imread(images[4])
            img2 = cv2.imread(images[iter-25])

            matches, kp1, kp2, mc1, mc2 = extract_and_match(img1, img2, Three = False, Four = True)
            np.save(f"two_view_recon_info/sfm{iter-25}_matched_idx.npy", mc1)

            inlinear = np.load("two_view_recon_info/inlinear.npy") # 4번 이미지 기준으로 R,t를 근사할 것이기 때문에 사용가능
            points_3d = np.load("two_view_recon_info/3D_points.npy")

            p1 = None
            p2 = np.load("two_view_recon_info/sfm04_camera_pose.npy")
        else : 
            try:
                # 나머지는 근사된 Pose들이 있으므로 Bundle Adjustment를 믿고 랜덤 샘플
                # 3, 4이미지를 제외 
                idx = np.random.randint(0, 14, 2)
                while any(value in [3,4] for value in idx):
                    idx = np.random.randint(0, 14, 2)
                
                img1 = cv2.imread(images[idx[0]])
                img2 = cv2.imread(images[idx[1]]) 

                matches, kp1, kp2, mc1, mc2 = extract_and_match(img1, img2, Three = False, Four = False)

                np.save(f"two_view_recon_info/sfm{idx[0]}_matched_idx.npy", mc1)
                np.save(f"two_view_recon_info/sfm{idx[0]}_matched_idx.npy", mc2)

                ### TODO 
                # inlinear를 만들어야함 + 3D Points도.
                # 레퍼런스 이미지는 무조건 2번 이미지 (iter=1 제외)

                inlinear = None # 2번째 이미지 기준으로 R,T를 근사하기 위해 2번째 이미지의 inlinear를 불러와야함 
                points_3d = None

                p1 = None 
                p2 = None # 레퍼런스 이미지인 2번째 이미지에 대한 p2를 저장해서 쓸 수 있으면 좋을듯?

            except :
                break

        # 3-point RANSAC
        R, T, p1, p2, cor, cor_2, cor_3, inlinear_out = rancsac(
            matches,
            kp1, 
            kp2, 
            mc1, 
            mc2,  
            k, 
            kinv,
            p1,
            p2,
            inlinear,
            points_3d,)
        
        # save inliers
        np.save(f"temp_result/inlinear_{iter}.npy", inlinear_out)

        # reconstruct 3D points
        points3D, threeDinof = Traiangulation(
            matches, # matching된 포인트로 inlinear를 만들어 주기  **** TODO**** but mc1, mc2가 있고 ransac 첫부분에서 만들어 주므로 check해보고 이걸 이용해서 traingulation 짜보기 
            kp1, 
            kp2,  
            R, 
            T, 
            p1, 
            p2,
            cor, 
            cor_2, 
            cor_3,
            k, 
            kinv,
            inlinear_out,
            threeDinof,)
        
        # save 3D points
        np.save(f"temp_result/3D_points_{i}.npy", points3D)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3D)
        o3d.io.write_point_cloud(f"result/3D_points_{iter}.ply", pcd)

        threedpoints.append(points3D)

    for i in threeDinof.keys():
        threeDpoint = threeDinof[i]["3D"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(threeDpoint)
        o3d.io.write_point_cloud(f"result/final/3D_points_{iter}.ply", pcd)

    ### TODO
    # 3D points 저장할 때 딕셔너리에서 조회해서 거리를 측정하면 될거같음 

if __name__ == '__main__':
    main()