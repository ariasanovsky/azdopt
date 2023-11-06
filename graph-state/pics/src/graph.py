import sys

import matplotlib.pyplot as plt
import networkx as nx
import imageio.v2 as imageio
import math

import matplotlib.pyplot as plt
import numpy as np

from .data import Data

from networkx import Graph

# todo! add a function to plot graphs with costs
def graph_gif(
    data: list['Data'],
    time_step: int,
    num_time_steps: int,
):
    frames = []
    total_num_steps = 0

    time_passed = 0

    for datum in data:
        graph = datum.graph()
        
        # plot the graph one time
        plt.figure()
        nx.draw_circular(
            graph,
            with_labels = True,
            # node_color = "white",
        )
        # plt.title(f"Cost: {datum.cost}")
        plt.text(
            0.02,
            0.08,
            f"Time: {time_passed}\nCost: {datum.cost}",
            transform=plt.gca().transAxes
        )
        
        # also plot the current time and cost
        plt.savefig("temp.png")
        frames.append(imageio.imread("temp.png"))
        plt.close()
        total_num_steps += 1

        # now with d = duration, we plot d / time_step times
        for t in range(0, datum.duration, time_step):
            dummy_time_passed = (time_passed + t) // time_step * time_step
            plt.figure()
            nx.draw_circular(
                graph,
                with_labels = True,
                # node_color = "white",
            )
            plt.text(
                0.02,
                0.08,
                f"Time: {dummy_time_passed}\nCost: {datum.cost}",
                transform=plt.gca().transAxes
            )
            # also plot the current time
            plt.savefig("temp.png")
            frames.append(imageio.imread("temp.png"))
            plt.close()

            total_num_steps += 1
            if total_num_steps >= num_time_steps:
                print("Reached max number of time steps")
                sys.exit(1)
        time_passed += datum.duration
    
    # pad up to the number of time steps
    for i in range(total_num_steps, num_time_steps):
        plt.figure()
        nx.draw_circular(
            graph,
            with_labels = True,
            # node_color = "white",
        )
        plt.text(
            0.02,
            0.08,
            f"Cost: {datum.cost}\nTime: {time_passed}",
            transform=plt.gca().transAxes
        )
        # also plot the current time
        plt.savefig("temp.png")
        frames.append(imageio.imread("temp.png"))
        plt.close()
    
    # delete the temp file
    import os
    os.remove("temp.png")

    imageio.mimsave(
        "out.gif",
        frames,
        duration = num_time_steps,
    )