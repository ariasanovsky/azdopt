import os
import matplotlib.pyplot as plt
import numpy as np

from .data import Data

from mpl_toolkits.mplot3d import Axes3D

'''
input: `list['Data']`
output: coordinates of a line segment supporting the corresponding stepfunction

example:
input = [
    Data(short_form='I', cost=Cost(matching=[{'max': 5, 'min': 2}], lambda_1=5.0), duration=2),
    Data(short_form='I', cost=Cost(matching=[], lambda_1=5.5), duration=3),
]
output = [
    (0, 6), (2, 6),
            (2, 5.5), (5, 5.5),
]
'''

def stepfunction_coordinates(
    data_list: list['Data'],
    _3d: bool = False,
    offset: int = 0,
) -> list:
    t: int = offset
    if _3d:
        coordinates: list[(int, float, float)] = []
        # import sys
        # sys.exit(1)
        for data in data_list:
            # print(f"data: {data}")
            # print(f"data.cost: {data.cost}")
            # print(f"data.cost.cost(): {data.cost.cost()}")
            # print(f"data.duration: {data.duration}")
            # sys.exit(1)
            m, l = data.cost.cost_2d()
            coordinates.append((t, m, l))
            t += data.duration
            coordinates.append((t, m, l))
        return coordinates
    else:
        coordinates: list = []
        for data in data_list:
            # print(f"data: {data}")
            # print(f"data.cost: {data.cost}")
            # print(f"data.cost.cost(): {data.cost.cost()}")
            # print(f"data.duration: {data.duration}")
            # sys.exit(1)
            coordinates.append((t, data.cost.cost()))
            t += data.duration
            coordinates.append((t, data.cost.cost()))
        return coordinates

def plot_costs_3d(
    data: list[list[list['Data']]],
    dir_name: str,
    batch_size: int,
    num_colors: int = 8,
    dir_name_trim_start: int = 39,
    dir_name_trim_end: int = 16,
):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for epoch, sublist in enumerate(data):
        for batch_i, data_list in enumerate(sublist):
            coordinates = stepfunction_coordinates(
                data_list,
                _3d = True,
                offset = epoch * 800,
            )
            t, y, z = zip(*coordinates)
            ax.plot(t, y, z, color=colors[batch_i % num_colors])
    ax.set_title(f'[{batch_size} batches] {dir_name[dir_name_trim_start:-dir_name_trim_end]}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Matching')
    ax.set_zlabel('Lambda')
    # save to dir_name/3d_cost.png
    plt.savefig(os.path.join(dir_name, '3d_cost.png'))
            

def new_plot_costs(
    data: list[list[list['Data']]],
    dir_name: str,
    batch_size: int,
    num_colors: int = 8,
    dir_name_trim_start: int = 39,
    dir_name_trim_end: int = 16,
    episodes_per_epoch: int = 800,
):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure()
    for epoch, sublist in enumerate(data):
        for batch_i, data_list in enumerate(sublist):
            coordinates = stepfunction_coordinates(
                data_list,
                offset = epoch * episodes_per_epoch,
            )
            x, y = zip(*coordinates)
            # print(f"coordinates: {coordinates}")
            # sys.exit(1)
            # x = [coordinate[0] for coordinate in coordinates]
            # y = [coordinate[1] for coordinate in coordinates]
            plt.plot(x, y, color=colors[batch_i % num_colors])
    plt.title(f'[{batch_size} batches] {dir_name[dir_name_trim_start:-dir_name_trim_end]}')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.legend([f'{i} % ({num_colors})' for i in range(num_colors)], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, 'new_cost.png'))
    plt.close()


def plot_costs(
    data: list[list[list['Data']]],
    dir_name: str,
    batch_size: int,
    num_colors: int = 8,
    dir_name_trim_start: int = 39,
    dir_name_trim_end: int = 16,
):
    x_sums = np.zeros(batch_size)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure()
    for epoch, sublist in enumerate(data):
        for batch_i, data_list in enumerate(sublist):
            y = [data.cost.cost() for data in data_list]
            x = np.cumsum([data.duration for data in data_list])
            plt.step(
                [_x + x_sums[batch_i] for _x in x],
                y,
                where='pre',
                color=colors[batch_i % num_colors],
            )
            x_sums[batch_i] += x[-1]
    plt.title(f'[{batch_size} batches] {dir_name[dir_name_trim_start:-dir_name_trim_end]}')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    # plt.legend([f'Batch {i}' for i in range(batch_size)])
    plt.legend([f'{i} % ({num_colors})' for i in range(num_colors)], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # save plot to file in the directory
    plt.savefig(os.path.join(dir_name, 'cost.png'))
    plt.close()
