import numpy as np
import cv2

INF = 999999999


def Birchfield_Tomasi_dissimilarity(left_image, right_image, d):
    # TODO: Implement Birchfield-Tomasi dissimilarity
    # Hint: Fill undefined elements with np.inf at the end

    raise NotImplementedError(
        "Birchfield_Tomasi_dissimilarity function has not been implemented yet")

    left_cost_volume = None
    right_cost_volume = None

    left_disparity = left_cost_volume.argmin(axis=2)
    right_disparity = right_cost_volume.argmin(axis=2)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity


def SAD(left_image, right_image, d):
    """
    i) image는 일단 gray scale로 받아옴
    ii) 
    """
    global INF

    left_cost_volume = np.full(
        (d, 215, 328), INF)  # Max value로 full array 생성
    right_cost_volume = np.full((d, 215, 328), INF)

    for i in range(d):
        disparity = left_image[:, i:] - right_image[:, : 328-i]
        left_cost_volume[i, :, i:] = abs(disparity)

    for i in range(d):
        disparity = right_image[:, i:] - left_image[:, : 328-i]
        right_cost_volume[i, :, :328-i] = abs(disparity)

    # left_cost_volume = None
    # right_cost_volume = None

    left_disparity = left_cost_volume.argmin(axis=0)
    right_disparity = right_cost_volume.argmin(axis=0)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity


def SSD(left_image, right_image, d):

    global INF

    # Max value로 full array 생성
    left_cost_volume = np.full((d, 215, 328), INF)
    right_cost_volume = np.full((d, 215, 328), INF)

    for i in range(d):
        disparity = left_image[:, i:] - right_image[:, : 328-i]
        left_cost_volume[i, :, i:] = np.square(disparity)

    for i in range(d):
        disparity = right_image[:, i:] - left_image[:, : 328-i]
        right_cost_volume[i, :, :328-i] = np.square(disparity)

    # left_cost_volume = None
    # right_cost_volume = None

    left_disparity = left_cost_volume.argmin(axis=0)
    right_disparity = right_cost_volume.argmin(axis=0)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity


"""### test code
image_c = cv2.imread('input/04_noise25.png',0)
image_l = cv2.imread('input/03_noise25.png',0)

left_cost_volume, right_cost_volume, left_disparity, right_disparity = SAD(image_l, image_c, 24)
print(left_cost_volume.shape)"""
