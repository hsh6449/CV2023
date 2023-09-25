import numpy as np
import sys
from tqdm import tqdm

MEMO = -np.ones((24, 215, 328))
INF = 999999999

sys.setrecursionlimit(10**9)


def aggregate_cost_volume(cost_volume, d=24):

    # TODO: Implement cost volume aggregation
    # cost_volume : (24, 215, 328) (Depth, Height, Width)
    # Either Left or Right, It doesn't matter.
    global INF
    global MEMO

    aggregated_costs = np.full(cost_volume.shape, INF)

    forward_pass = generate_cor_pass(cost_volume, 1)  # size조심 (window 크기)
    backward_pass = generate_cor_pass(cost_volume, 1)

    aggregated_volume = []

    forward_direction = [(0, 1), (-1, 1), (-1, 0),
                         (-1, -1)]  # (y, x) 좌, 좌위, 위, 우위
    backward_direction = [(0, -1), (1, -1), (1, 0),
                          (1, 1)]  # (y, x) 우, 우아래, 아래, 좌아래

    for r in forward_direction:

        MEMO = np.full((24, 215, 328), INF)  # 메모 초기화

        for k in tqdm(range(d)):
            for _, (dy, dx) in enumerate(forward_pass):

                # cost volume shape : (24, 215, 328)
                # forward_pass shape : (dy, dx); (215, 328)

                # DONE ; Implement SGM - idx issue 해결, Implement SGM - sort 구현함 , 방향(r) 구현

                # forward : r; (y,x)= 좌 좌위 위 우위
                # backward : r; (y,x)= 우 우아래 아래 좌아래
                aggregated_costs[k, dy, dx] = SGM(cost_volume, dy, dx, k, r=r)
        aggregated_volume.append(aggregated_costs)
        # print(aggregated_costs)

    for r in backward_direction:

        MEMO = np.full((24, 215, 328), INF)  # 메모 초기화

        for k in tqdm(range(d)):
            for _, (dy, dx) in reversed(list(enumerate(backward_pass))):
                # TODO: Implement backward pass
                aggregated_costs[k, dy, dx] = SGM(
                    cost_volume, dy, dx, 23-k, r=r)
        aggregated_volume.append(aggregated_costs)
        # print(aggregated_costs.shape)

    aggregated_volume = np.sum(aggregated_volume, axis=0)
    # print("aggregated_volume : ", aggregated_volume.shape)
    return aggregated_volume


def SGM(cost_volume, dy, dx, d, r, p1=5, p2=150, size=1):

    global MEMO
    global INF

    if (dy < 0) | (dx < 0) | (dy > 214) | (dx > 327) | (d < 0) | (d > 23):
        cost = INF
        MEMO[d, dy, dx] = cost
        return cost

    # elif (dy == 0) & (dx == 0):
    #     cost = cost_volume[d, dy, dx]
    #     MEMO[d, dy, dx] = cost
    #     return cost
    # elif (d < 0) | (d > 23):
    #     cost = INF
    #     return cost

    else:
        if check_memo(d, dy, dx) != INF:
            cost = MEMO[d, dy, dx]
            return cost

        min_value_list = []
        for i in range(24-d):
            temp = check_memo(i, dy-r[0], dx-r[1])
            min_value_list.append(temp)

        min_value = quick_sort(min_value_list)

        values = []
        for k in range(24):
            values.append(check_memo(k, dy-r[0], dx-r[1]))

        cost = (cost_volume[d, dy, dx] + quick_sort([check_memo(d, dy-r[0], dx-r[1]), check_memo(d-1, dy-r[0],
                                                                                                 dx-r[1])+p1, check_memo(d+1, dy-r[0], dx-r[1])+p1, min_value+p2]) - quick_sort(values))

        # quick_sort([SGM(cost_volume, dy-r[0], dx-r[1], d), SGM(cost_volume,dy-r[0], dx-1-r[1], d) + p1, SGM(cost_volume, dy-r[0], dx+1-r[1], d) + p1, min_value + p2]) - quick_sort(MEMO[:, dy-r[0], dx-r[1]]))
        MEMO[d, dy, dx] = cost

    return cost


def quick_sort(array):

   # new_array = array.copy()

    if len(array) <= 1:
        return array[0]

    else:
        pivot = array[round(len(array) // 2)]

        less = []
        # equal = []
        # greater = []

        for i in array:
            if i < pivot:
                less.append(i)
            else:
                pass
        if len(less) == 0:
            less.append(pivot)
        return quick_sort(less)


def generate_cor_pass(cost_volume, size=1):

    window = []

    x = cost_volume.shape[2]/size
    y = cost_volume.shape[1]/size

    for i in range(int(y)):
        for j in range(int(x)):
            window.append((i, j))  # i 가 y, j 가 x
    return window


def check_memo(d, dy, dx):
    global MEMO
    global INF

    if 0 <= d < 24 and 0 <= dy < 215 and 0 <= dx < 328:
        return MEMO[d, dy, dx]
    else:
        return INF
