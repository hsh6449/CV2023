import numpy as np
from tqdm import tqdm


def warp_image(image, disparity_map, loc='left'):
    '''
    Image warping

    image : Original image with channel (H, W, 3)
    disparity_map : Disparity map (H, W)
    loc : Location of image (left or right) compare to reference image
    '''

    # Image와 같은 크기의 warped_image 생성
    warp = image.copy()

    # left일 경우 와 right일 경우 나눠서 처리
    if loc == "left":

        # disparity는 좌표를 나타내기 때문에 image의 해당 좌표를 조회해서 값을 넣어줌
        for i in range(215):
            for j in range(328):
                if j+disparity_map[i, j] > 327:  # index가 넘어갈 경우 예외처리
                    warp[i, j] = 0
                else:
                    warp[i, j] = image[i, j+disparity_map[i, j]]

    elif loc == "right":
        for i in range(215):
            for j in range(328):
                if j-disparity_map[i, j] < 0:  # index가 넘어갈 경우 예외처리
                    warp[i, j] = 0
                else:
                    warp[i, j] = image[i, j-disparity_map[i, j]]

    return warp
