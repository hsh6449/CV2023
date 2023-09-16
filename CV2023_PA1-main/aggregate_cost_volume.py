import numpy as np
from tqdm import tqdm


def aggregate_cost_volume(cost_volume):
    # TODO: Implement cost volume aggregation

    ## 1. left cost volume과 right cost volume을 구별 해야함 
    ## 2. cost volume을 구할때 겹치는 부분만 계산해야하는데, max depth는 24
    ## 3. 
    cost_volume

    aggregated_costs = None

    forward_pass = list()
    backward_pass = list()

    for idx, (dy, dx) in enumerate(forward_pass):
        # TODO: Implement forward pass
        pass

    for idx, (dy, dx) in enumerate(backward_pass):
        # TODO: Implement backward pass
        pass

    aggregated_volume = np.sum(aggregated_costs, axis=3)
    return aggregated_volume
