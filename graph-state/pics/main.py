from src.data import Data
from src.plot_costs import plot_costs, new_plot_costs, plot_costs_3d

import sys

from src.graph import graph_gif

def main(dir_name: str, batch_size: int):
    all_data: list[list['Data']] = Data.read_data(dir_name, batch_size)
    # new_plot_costs(all_data, dir_name, batch_size)
    plot_costs_3d(all_data, dir_name, batch_size)
    graph_gif(all_data[0][0], 10, 120)
    return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python de.py [DIR] [NUM_THREADS]")
        sys.exit(1)
    main(
        str(sys.argv[1]),
        int(sys.argv[2]),
    )
