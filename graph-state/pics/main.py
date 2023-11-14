import argparse

from src.data import Data
from src.plot_costs import plot_costs, new_plot_costs, plot_costs_3d
from src.arguments import Arguments

import sys

from src.graph import graph_gif

def main(
    dir_name: str,
    batch_size: int,
    dim: int,
    tree: bool = False,
):
    all_data: list[list['Data']] = Data.read_data(
        dir_name,
        batch_size,
    )
    # if tree:
    #     print("Tree is unimplemented")
    #     sys.exit(1)
    # new_plot_costs(all_data, dir_name, batch_size)
    plot_costs_3d(all_data, dir_name, batch_size)
    # graph_gif(all_data[0][0], 10, 120)
    return None

if __name__ == "__main__":
    args: Arguments = Arguments.from_json('default_args.json')
    print(f"defaults:         {args}")
    # all arguments are optional overrides with no default values
    parser = argparse.ArgumentParser(description = "For plotting output data")
    parser.add_argument('-dir', type = str, help = 'Directory')
    parser.add_argument('-p',   type = str, help = 'Project name')
    parser.add_argument('-b',   type = int, help = 'Batch size')
    parser.add_argument('-dim', type = int, choices = [2, 3], help = 'Dimension')
    parser.add_argument('-t',   action = 'store_true' , help = 'Tree flag')
    
    # override defaults
    overrides = parser.parse_args() # namespace = args)
    args.override_with(overrides)
    print(f"after overrides:  {args}")
    # assign last project
    if args.project_name is None:
        args.assign_last_project()
    print(f"after assignment: {args}")
    main(
        dir_name = args.project_path(),
        batch_size = args.num_blocks,
        dim = args.dim,
        tree = args.tree,
    )
