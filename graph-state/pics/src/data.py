import os
import sys

import json

import glob

from networkx import from_graph6_bytes, Graph

class Data:
    def __init__(self, short_form: str, cost, duration: int):
        self.short_form = short_form
        self.cost = cost
        self.duration = duration

    def __repr__(self):
        return f"Data(short_form={self.short_form}, cost={self.cost}, duration={self.duration})"

    def graph(self) -> Graph:
        return from_graph6_bytes(self.short_form.encode('ascii'))
    
    @staticmethod
    def read_data(dir_name: str, batch_size: int) -> list[list['Data']]:
        # Find all JSON files in the directory
        json_files: list[str] = glob.glob(os.path.join(dir_name, '*.json'))
        json_files.sort()
        data_lists: list[list['Data']] = []
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data_list: list[list[dict]] = json.load(f)
                # Convert dictionaries to Data objects
                if len(data_list) != batch_size:
                    print(f"Error: {file_path} has {len(data_list)} threads")
                    sys.exit(1)
                data_list: list = [[Data(**data_dict) for data_dict in sublist] for sublist in data_list]
                data_lists.append(data_list)
        return data_lists
