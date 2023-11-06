import os
import sys

import json
import matplotlib.pyplot as plt
import numpy as np

import glob

# import networkx as nx
# from networkx import from_graph6_bytes

class Data:
    def __init__(self, short_form, cost, duration):
        self.short_form = short_form
        self.cost = cost
        self.duration = duration

    def __repr__(self):
        return f"Data(short_form={self.short_form}, cost={self.cost}, duration={self.duration})"

def main(dir_name, num_threads):
    json_files = glob.glob(os.path.join(dir_name, '*.json'))
    json_files.sort()
    x_sums = np.zeros(num_threads)
    # assign each thread a distinct using the same operation that `plt` uses to determine distinct colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure()
    for epoch, file_path in enumerate(json_files):
        if not file_path.endswith(f'epoch{epoch}.json'):
            print(f"Error: {file_path} is not epoch{epoch}.json")
            sys.exit(1)
        with open(file_path, 'r') as f:
            data_list = json.load(f)
            if len(data_list) != num_threads:
                print(f"Error: {file_path} has {len(data_list)} threads")
                sys.exit(1)
            for thread_i, sublist in enumerate(data_list):
                costs = []
                durations = []
                for _, data_dict in enumerate(sublist):
                    data = Data(**data_dict)
                    costs.append(data.cost)
                    durations.append(data.duration)
                y = costs
                x = np.cumsum(durations)
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
    plt.legend([f'Thread {i}' for i in range(num_threads)])
    # save plot to file in the directory
    plt.savefig(os.path.join(dir_name, 'cost.png'))
    plt.close()
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python de.py [DIR] [NUM_THREADS]")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
