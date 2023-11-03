import networkx as nx
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from networkx import from_graph6_bytes

import sys
import os

def plot_graph(graph, filename):
    nx.draw_circular(graph, with_labels=True)
    plt.savefig(filename)
    plt.close()

def create_gif(images, output):
    with imageio.get_writer(output, mode='I') as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)

def main(graph6_file, output_dir):
    images = []
    graph6_file = os.path.join(output_dir, graph6_file)
    with open(graph6_file, 'r') as file:
        for i, line in enumerate(file):
            graph6 = line.strip()
            graph = from_graph6_bytes(graph6.encode())
            filename = os.path.join(output_dir, f"graph{i}.png")
            plot_graph(graph, filename)
            images.append(filename)
    gif_filename = os.path.join(output_dir, "graphs.gif")
    create_gif(images, gif_filename)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot_graphs.py <output_dir>")
        sys.exit(1)
    output_dir = sys.argv[1]
    main("out.g6", output_dir)
