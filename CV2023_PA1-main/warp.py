import numpy as np
from tqdm import tqdm


def warp_image(image, disparity_map, loc='left'):
    # TODO: Implement image warping
    # print(disparity_map.shape)
    warp = image.copy()
    if loc == "left":
        for i in range(215):
            for j in range(328):
                # image는 (215,328, 3)
                warp[i, j] = image[i, j+disparity_map[i, j]]
    elif loc == "right":
        for i in range(215):
            for j in range(328):
                warp[i, j] = image[i, j-disparity_map[i, j]]
    # warped_image = warp

    return warp
