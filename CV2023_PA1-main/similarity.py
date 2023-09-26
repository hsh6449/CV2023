import numpy as np
import cv2

INF = 99999  # Set Infinte value


def SAD(left_image, right_image, d):
    """
    Sum of Absolute Differences

    left_image : gray scale image (H, W)
    right_image : Reference gray scale image (H, W)
    d : Depth

    """
    global INF

    # INF 로 Cost Volume과 shape이 같은 array 생성 (left, right 둘 다)
    # Expected Cost Volume의 shape은 (d, H, W), 이번 과제의 경우는 (24, 215, 328)

    left_cost_volume = np.full(
        (d, 215, 328), INF)
    right_cost_volume = np.full((d, 215, 328), INF)

    for i in range(d):

        # 왼쪽에서 오른쪽으로 이동하면서 disparity 계산
        disparity_l = left_image[:, i:] - right_image[:, : 328-i]
        left_cost_volume[i, :, i:] = abs(disparity_l)

        # 오른쪽에서 왼쪽으로 이동하면서 disparity 계산
        disparity_r = right_image[:, i:] - left_image[:, : 328-i]
        right_cost_volume[i, :, :328-i] = abs(disparity_r)

    # SAD를 이용해 계산한 Disparity
    left_disparity = left_cost_volume.argmin(axis=0)
    right_disparity = right_cost_volume.argmin(axis=0)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity


def SSD(left_image, right_image, d):
    """
    Sum of Square Differences

    left_image : gray scale image (H, W)
    right_image : Reference gray scale image (H, W)
    d : Depth

    """

    global INF

    # INF 로 Cost Volume과 shape이 같은 array 생성 (left, right 둘 다)
    left_cost_volume = np.full((d, 215, 328), INF)
    right_cost_volume = np.full((d, 215, 328), INF)

    for i in range(d):

        # 왼쪽에서 오른쪽으로 이동하면서 disparity 계산
        disparity_l = left_image[:, i:] - right_image[:, : 328-i]
        left_cost_volume[i, :, i:] = np.square(disparity_l)

        # 오른쪽에서 왼쪽으로 이동하면서 disparity 계산
        disparity_r = right_image[:, i:] - left_image[:, : 328-i]
        right_cost_volume[i, :, :328-i] = np.square(disparity_r)

    # SSD를 이용해 계산한 Disparity
    left_disparity = left_cost_volume.argmin(axis=0)
    right_disparity = right_cost_volume.argmin(axis=0)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity


"""### test code
image_c = cv2.imread('input/04_noise25.png',0)
image_l = cv2.imread('input/03_noise25.png',0)

left_cost_volume, right_cost_volume, left_disparity, right_disparity = SAD(image_l, image_c, 24)
print(left_cost_volume.shape)"""
