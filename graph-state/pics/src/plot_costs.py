import os
import matplotlib.pyplot as plt
import numpy as np

from .data import Data

def plot_costs(
    data: list[list[list['Data']]],
    dir_name: str,
    batch_size: int,
):
    x_sums = np.zeros(batch_size)
    # assign each thread a distinct using the same operation that `plt` uses to determine distinct colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure()
    for epoch, sublist in enumerate(data):
        for thread_i, data_list in enumerate(sublist):
            y = [data.cost for data in data_list]
            x = np.cumsum([data.duration for data in data_list])
            plt.step(
                [_x + x_sums[thread_i] for _x in x],
                y,
                where='pre',
                color=colors[thread_i]
            )
            x_sums[thread_i] += x[-1]
    plt.title('Cost')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.legend([f'Thread {i}' for i in range(batch_size)])
    # save plot to file in the directory
    plt.savefig(os.path.join(dir_name, 'cost.png'))
    plt.close()
