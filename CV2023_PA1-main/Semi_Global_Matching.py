import cv2
import numpy as np
from tqdm import tqdm
import os

from similarity import Birchfield_Tomasi_dissimilarity, SAD
from aggregate_cost_volume import aggregate_cost_volume
from warp import warp_image


# Modify any parameters or any function itself if necessary.
# Add comments to parts related to scoring criteria to get graded.

def semi_global_matching(left_image, right_image, d):
    # TODO: Implement Semi-Global Matching

    raise NotImplementedError("Semi_Global_Matching function has not been implemented yet")

    left_cost_volume, right_cost_volume, left_disparity, right_disparity = Birchfield_Tomasi_dissimilarity(left_image, right_image, d)

    # TODO: save cost disparity

    cost_volume = None
    aggregated_cost_volume = aggregate_cost_volume(cost_volume)
    aggregated_disparity = aggregated_cost_volume.argmin(axis=2)

    # TODO: save Semi Global Matching disparity


if __name__ == "__main__":
    img_list = list()
    ground_truth = None
    # TODO: Load required images

    d = 24

    for i in range(len(img_list)):
        # TODO: Perform Semi-Global Matching
        pass

    warped_image_list = list()
    for i, image in enumerate(img_list):
        # TODO: Warp image
        pass

    boundary_range = d
    cropped_ground_truth = ground_truth[boundary_range:-boundary_range, boundary_range:-boundary_range]

    # TODO: Aggregate warped images

    # TODO: Compute MSE and PSNR
    mse = np.inf
    print("mse: {mse}".format(mse=mse))

    psnr = np.inf
    print("psnr: {psnr}".format(psnr=psnr))

    # TODO: Save aggregated disparity
