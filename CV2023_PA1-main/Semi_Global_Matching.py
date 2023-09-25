import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import glob
import time

from similarity import Birchfield_Tomasi_dissimilarity, SAD, SSD
from aggregate_cost_volume import aggregate_cost_volume
from warp import warp_image


# Modify any parameters or any function itself if necessary.
# Add comments to parts related to scoring criteria to get graded.

def semi_global_matching(left_image, right_image, d):

    # left_cost_volume, right_cost_volume, left_disparity, right_disparity = Birchfield_Tomasi_dissimilarity(left_image, right_image, d)
    left_cost_volume_, right_cost_volume_, left_disparity, right_disparity = SAD(
        left_image, right_image, d)
    # print("cost volume : ", left_cost_volume_.shape)

    left_cost_volume = aggregate_cost_volume(left_cost_volume_)
    right_cost_volume = aggregate_cost_volume(right_cost_volume_)

    # print("aggregated cost volume : ", left_cost_volume.shape)

    aggregated_cost_volume = np.sum(
        [left_cost_volume, right_cost_volume], axis=0)
    aggregated_disparity = aggregated_cost_volume.argmin(axis=0)

    return aggregated_disparity


if __name__ == "__main__":

    img_list = [cv2.imread(file, -1)
                for file in glob.glob("input/*.png")]  # 원본이미지
    mod_list = [cv2.imread(file, 0)
                for file in glob.glob("input/*.png")]  # disparity 계산을 위한 gray scale 이미지

    ground_truth = cv2.imread("target/gt.png", -1)
    # TODO: Load required images

    d = 24

    agg_dis_list = []

    for i in range(len(mod_list)):
        # TODO: Perform Semi-Global Matching
        # 여기도 아닐 수도 있음 다시 생각
        if i < 3:
            aggregated_disparity = semi_global_matching(
                mod_list[i], mod_list[3], d)
            agg_dis_list.append(aggregated_disparity)
        else:  # i > 3
            aggregated_disparity = semi_global_matching(
                mod_list[3], mod_list[i], d)
            agg_dis_list.append(aggregated_disparity)

    warped_image_list = list()
    for i, image in enumerate(img_list):
        # TODO: Warp image
        if i < 3:
            warped_image = warp_image(image, agg_dis_list[i], loc='left')
            warped_image_list.append(warped_image)
        else:  # i > 3
            warped_image = warp_image(image, agg_dis_list[i], loc='right')
            warped_image_list.append(warped_image)

    boundary_range = d
    cropped_ground_truth = ground_truth[boundary_range:-
                                        boundary_range, boundary_range:-boundary_range]

    # TODO: Aggregate warped images
    for i, img in enumerate(warped_image_list):
        cv2.imwrite(
            f"output/Intermediate_Disparity/warped_image_{i}_{time.time()}.png", img)

    aggregated_warped_image = np.mean(warped_image_list, axis=0)
    cv2.imwrite(
        f"output/agg_warped_image_{time.time()}.png", aggregated_warped_image)

    # TODO: Compute MSE and PSNR
    mse = np.sum(np.square(ground_truth - aggregated_warped_image)) / \
        (img_list[0].shape[0]*img_list[0].shape[1]*img_list[0].shape[2])
    print("mse: {mse}".format(mse=mse))

    psnr = 10 * np.log10(255**2 / mse)
    print("psnr: {psnr}".format(psnr=psnr))

    cv2.imshow("agg_warped_image", aggregated_warped_image.astype(np.uint8))
    cv2.waitKey(0)

    # TODO: Save aggregated disparity
    # 에러 존재
    # np.save(
    #     f"output/Final_Disparity/agg_disparity_{time}.npy", aggregated_disparity)

    # Save cost
    # cost = f"mse: {mse}, psnr: {psnr}"
    # file = open(f"output/Cost/argmin cost volume.txt", "w")
    # file.write(cost)
    # file.close()
