import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import glob
import time

from similarity import SAD, SSD
from aggregate_cost_volume import aggregate_cost_volume
from warp import warp_image


# Modify any parameters or any function itself if necessary.
# Add comments to parts related to scoring criteria to get graded.

def semi_global_matching(left_image, right_image, d, direction='left'):

    # SAD를 이용하여 cost volume을 구함
    left_cost_volume_, right_cost_volume_, left_disparity, right_disparity = SAD(
        left_image, right_image, d)

    # aggregate cost volume을 구함 (left, rigt)
    if direction == 'left':
        aggregated_cost_volume = aggregate_cost_volume(
            left_cost_volume_)  # (24, 215, 328)

    else:  # right
        aggregated_cost_volume = aggregate_cost_volume(right_cost_volume_)

    # 만들어진 aggregated cost volume을 이용하여 disparity를 구함 (argmin)
    aggregated_disparity = aggregated_cost_volume.argmin(axis=0)
    return aggregated_disparity  # (215, 328)


if __name__ == "__main__":

    img_list = [cv2.imread(file, -1)
                for file in glob.glob("input/*.png")]  # 원본이미지
    mod_list = [cv2.imread(file, 0)
                for file in glob.glob("input/*.png")]  # disparity 계산을 위한 gray scale 이미지

    ground_truth = cv2.imread("target/gt.png", -1)  # ground truth 이미지

    d = 24  # Depth

    agg_dis_list = []  # aggregated disparity를 저장할 list

    for i in range(len(mod_list)):  # disparity 계산을 위한 gray scale 이미지를 이용

        if i < 3:  # left image case, 4번째 이미지가 reference image
            aggregated_disparity = semi_global_matching(
                mod_list[i], mod_list[3], d, 'right')
            cv2.imwrite(
                f"/output/Intermediate_Disparity/agg_disparity_{i}_{time.time()}.png", aggregated_disparity*int(255/24))
            agg_dis_list.append(aggregated_disparity)

        elif i > 3:   # i > 3, right image case
            aggregated_disparity = semi_global_matching(
                mod_list[3], mod_list[i], d, 'left')
            cv2.imwrite(
                f"output/Intermediate_Disparity/agg_disparity_{i}_{time.time()}.png", aggregated_disparity*int(255/24))
            agg_dis_list.append(aggregated_disparity)
        else:  # i == 3, reference image case
            agg_dis_list.append([None])

    warped_image_list = list()  # warped image를 저장할 list

    for i, image in enumerate(img_list):  # 원본 이미지를 이용하여 warped image를 구함

        if i < 3:  # i < 3, left image case
            warped_image = warp_image(image, agg_dis_list[i], loc='left')
            warped_image_list.append(warped_image)
        elif i == 3:  # i == 3, reference image case
            pass
        elif i > 3:  # i > 3, right image case
            warped_image = warp_image(image, agg_dis_list[i], loc='right')
            warped_image_list.append(warped_image)

    boundary_range = d
    cropped_ground_truth = ground_truth[boundary_range:-
                                        boundary_range, boundary_range:-boundary_range]

    # TODO: Aggregate warped images
    for i, img in enumerate(warped_image_list):
        cv2.imwrite(
            f"output/warped_images/warped_image_{i}_{time.time()}.png", img)

    aggregated_warped_image = np.mean(warped_image_list, axis=0)
    cropped_warped = aggregated_warped_image[boundary_range:-
                                             boundary_range, boundary_range:-boundary_range]
    cv2.imwrite(
        f"output/agg_warped_image_{time.time()}.png", aggregated_warped_image)

    # TODO: Compute MSE and PSNR
    mse = np.sum(np.square(cropped_ground_truth - cropped_warped)) / \
        (img_list[0].shape[0]*img_list[0].shape[1]*img_list[0].shape[2])
    print("mse: {mse}".format(mse=mse))

    psnr = 10 * np.log10(255**2 / mse)
    print("psnr: {psnr}".format(psnr=psnr))

    cv2.imshow("agg_warped_image", aggregated_warped_image.astype(np.uint8))
    cv2.waitKey(0)

    # TODO: Save aggregated disparity
    # 에러 존재
    np.save(
        f"output/Final_Disparity/agg_disparity_{time.time()}.npy", aggregated_disparity)
    cv2.imwrite(
        f"output/Final_Disparity/agg_disparity_{time.time()}.png", aggregated_disparity*int(255/24))

    # Save cost
    cost = f"mse: {mse}, psnr: {psnr}"
    file = open(f"output/Cost/argmin_cost_volume.txt", "w")
    file.write(cost)
    file.close()
