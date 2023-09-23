import numpy as np
from tqdm import tqdm

MEMO = -np.ones((24, 215, 328))
INF = 999999999


def aggregate_cost_volume(cost_volume, d=24):

    # TODO: Implement cost volume aggregation
    # cost_volume : (24, 215, 328) (Depth, Height, Width)
    # Either Left or Right, It doesn't matter.
    global INF
    global MEMO

    aggregated_costs = np.full(cost_volume.shape, INF)

    forward_pass = generate_cor_pass(cost_volume, 1)  # size조심 (window 크기)
    backward_pass = generate_cor_pass(cost_volume, 1)

    for k in tqdm(range(d)):
        for idx, (dy, dx) in enumerate(forward_pass):

            # cost volume shape : (24, 215, 328)
            # forward_pass shape : (dy, dx); (215, 328)

            # DONE ; Implement SGM - idx issue 해결, Implement SGM - sort 구현함
            # TODO : Implement SGM - 방향(r) 구현해야함

            aggregated_costs[k, dy, dx] = SGM(
                cost_volume, dy, dx, k, r=1)

    MEMO = -np.ones((24, 215, 328))

    ### *** 여기서 고민 *** ###
    # 1. forward pass랑 backward pass를 구현했는데 aggregated_cost에는 덮어 씌워지는거같음
    # 어차피 이따가 min을 취하는데 더 작은 값으로 덮어 씌우는게 맞는거 같음

    # 2. r을 일단 1로 설정해놨는데 이것이 8방향을 나타내지는 않기 때문에
    # 8방향을 나타내는 r을 구현해야함

    for k in tqdm(range(d)):
        for idx, (dy, dx) in reversed(list(enumerate(backward_pass))):
            # TODO: Implement backward pass
            temp_disparity = SGM(
                cost_volume, dy, dx, 23-k, r=1)
            if aggregated_costs[k, dy, dx] > temp_disparity:
                aggregated_costs[k, dy, dx] = temp_disparity
            else:
                pass
    aggregated_volume = np.sum(aggregated_costs, axis=0)
    return aggregated_volume


def SGM(cost_volume, dy, dx, d, r=1, p1=5, p2=100, size=1):

    global MEMO
    global INF

    if (dy <= 0) | (dx <= 0) | (dy >= 215) | (dx >= 328):
        cost = INF

    else:
        if MEMO[d, dy, dx] != -1:
            cost = MEMO[d, dy, dx]
            return cost

        for i in range(328-dx):
            temp = SGM(cost_volume, dy-r, i, d, r=1)
            if RecursionError:
                min_value = INF
            if min_value > temp:
                min_value = temp

        cost = (cost_volume[d, dy, dx] + quick_sort([SGM(cost_volume, dy-r, dx, d), SGM(cost_volume,
                                                                                        dy-r, dx-1, d) + p1, SGM(cost_volume, dy-r, dx+1, d) + p1, min_value + p2]) - quick_sort(MEMO[d, dy-r]))
        MEMO[d, dy, dx] = cost

    return cost


def quick_sort(array):

    new_array = array.copy()

    if len(new_array) <= 1:
        return new_array[0]

    else:
        pivot = new_array[round(len(new_array) // 2)]

        less = []
        # equal = []
        # greater = []

        for i in new_array:
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
