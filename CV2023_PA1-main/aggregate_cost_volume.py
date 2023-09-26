import numpy as np
import sys
from tqdm import tqdm

MEMO = -np.ones((24, 215, 328))  # Memoization for DP
INF = 999999999  # Set Infinte value


def aggregate_cost_volume(cost_volume, d=24):
    '''
    aggregate_cost_volume : Aggregate cost volume using Semi-Global Matching
    cost_volume : Cost volume (D, H, W)
    d : Depth
    '''
    # Either Left or Right, It doesn't matter.

    global INF  # Set Global value
    global MEMO  # Set Global value

    # Set Expected Cost Volume's shape (d, H, W)
    aggregated_costs = np.full(cost_volume.shape, INF)

    # Generate Coordinate information for each pixel (current state); custom function을 만듬
    forward_pass = generate_cor_pass(cost_volume, 1)
    backward_pass = generate_cor_pass(cost_volume, 1)

    # 각 방향별로 조사될 cost volume을 저장할 list
    aggregated_volume = []

    # forward pass의 방향과 backward pass의 방향을 나타내는 r을 만들어줌
    forward_direction = [(0, 1), (-1, 1), (-1, 0),
                         (-1, -1)]  # (y, x) 좌, 좌위, 위, 우위
    backward_direction = [(0, -1), (1, -1), (1, 0),
                          (1, 1)]  # (y, x) 우, 우아래, 아래, 좌아래

    # 방향을 기준으로 forward pass와 backward pass를 나눠서 진행
    for r in tqdm(forward_direction):

        MEMO = np.full((24, 215, 328), INF)  # 각 방향 마다 메모 초기화

        # for k in tqdm(range(d)):
        #     for _, (dy, dx) in enumerate(forward_pass):

        #         # cost volume shape : (24, 215, 328)
        #         # forward_pass shape : (dy, dx); (215, 328)

        #         aggregated_costs[k, dy, dx] = SGM(cost_volume, dy, dx, k, r=r)
        # aggregated_volume.append(aggregated_costs)
        for _, (k, dy, dx) in enumerate(forward_pass):
            aggregated_costs[k, dy, dx] = SGM(cost_volume, dy, dx, k, r=r)
        aggregated_volume.append(aggregated_costs)

    for r in tqdm(backward_direction):

        MEMO = np.full((24, 215, 328), INF)  # 메모 초기화

        # for k in tqdm(range(d)):
        #     for _, (dy, dx) in reversed(list(enumerate(backward_pass))):  # Backward는 반대로 진행
        #         aggregated_costs[k, dy, dx] = SGM(
        #             cost_volume, dy, dx, 23-k, r=r)
        # aggregated_volume.append(aggregated_costs)
        for _, (k, dy, dx) in reversed(list(enumerate(backward_pass))):
            aggregated_costs[k, dy, dx] = SGM(cost_volume, dy, dx, k, r=r)
        aggregated_volume.append(aggregated_costs)

    aggregated_volume = np.sum(aggregated_volume, axis=0)
    return aggregated_volume


def SGM(cost_volume, dy, dx, d, r, p1=5, p2=150, size=1):
    '''
    SGM : Semi-Global Matching
    cost_volume : Cost volume (D, H, W)
    dy : y coordinate of current state
    dx : x coordinate of current state
    d : Depth
    r : Direction
    p1 : Penalty 1
    p2 : Penalty 2
    size : Window size
    '''

    global MEMO
    global INF

    # Indexing을 벗어나서 값을 구할 수 없는 경우 예외 처리 (INF)

    if (dy < 0) | (dx < 0) | (dy > 214) | (dx > 327) | (d < 0) | (d > 23):
        cost = INF
        MEMO[d, dy, dx] = cost
        return cost

    else:
        # 이미 메모가 되어있는 경우 메모된 값을 반환
        if check_memo(d, dy, dx) != INF:
            return MEMO[d, dy, dx]

        # 현재 state의 cost volume에서의 값과 이전 state의 cost volume에서의 값들을 비교해서 최소값을 구함
        min_value_list = []
        for i in range(24):  # 이전버전으로 갈거면 328-d
            temp = check_memo(i, dy-r[0], dx-r[1])
            min_value_list.append(temp)
        min_value = quick_sort(min_value_list[d:])  # d부터 끝까지만 비교

        cost = (cost_volume[d, dy, dx] + quick_sort([check_memo(d-r[0], dy, dx-r[1]), check_memo(
            d-r[0], dy-1, dx-r[1])+p1, check_memo(d-r[0], dy+1, dx-r[1])+p1, min_value+p2]) - quick_sort(min_value_list))  # 전체 depth를 비교

        # 구한 cost를 메모에 저장
        MEMO[d, dy, dx] = cost

    return cost


def quick_sort(array):
    '''
    quick_sort : Quick sort algorithm (modified); 최적화를 위해 quick sort를 살짝 변형함
    array : List
    '''

    if len(array) <= 1:
        return array[0]

    else:
        pivot = array[round(len(array) // 2)]  # 중간값을 pivot으로 설정
        less = []  # 정렬이 목표가 아니라 최솟값을 찾는게 목표이므로 pivot보다 작은 값들만 저장

        for i in array:
            if i < pivot:
                less.append(i)
            else:
                pass

            # 만약 pivot보다 작은 값이 없다면 pivot을 반환 (값이 전부 같거나 pivot이 가장 작다고 판단)
            if len(less) == 0:
                less.append(pivot)
            return quick_sort(less)


def generate_cor_pass(cost_volume, size=1):
    '''
    generate_cor_pass : Generate coordinate information for each pixel (current state)
    cost_volume : Cost volume (D, H, W)
    size : Window size
    '''

    # current state의 좌표를 저장할 list
    window = []

    # cost volume의 shape만큼 반복해서 좌표를 정함
    x = cost_volume.shape[2]/size
    y = cost_volume.shape[1]/size

    for i in range(int(y)):
        for j in range(int(x)):
            for k in range(24):
                window.append((k, i, j))  # i 가 y, j 가 x

    return window


def check_memo(d, dy, dx):
    '''
    check_memo : Check memoization; 메모가 되어있는지 확인
    d : Depth
    dy : y coordinate of current state
    dx : x coordinate of current state
    '''
    global MEMO
    global INF

    # Index를 벗어나지 않으면 메모된 값을 반환
    if 0 <= d < 24 and 0 <= dy < 215 and 0 <= dx < 328:
        return MEMO[d, dy, dx]

    # Index를 벗어나면 INF를 반환
    else:
        return INF
