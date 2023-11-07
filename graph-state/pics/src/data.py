import os
import sys

import json

import glob

from networkx import from_graph6_bytes, Graph

class Cost:
    def __init__(self, cost: list):
        if len(cost) != 2:
            print(f"Error: Cost {cost} should have 2 elements")
            sys.exit(1)
        self.matching = cost[0]
        self.lambda_1 = cost[1]
        # print(f"self.matching: {self.matching}")
        # print(f"self.lambda_1: {self.lambda_1}")
        # sys.exit(1)
    
    def __repr__(self):
        return f"Cost(matching={self.matching}, lambda_1={self.lambda_1})"

    def cost(self) -> float:
        return len(self.matching) + self.lambda_1

    def cost_2d(self) -> (float, float):
        return (float(len(self.matching)), self.lambda_1)

class Data:
    def __init__(self, short_form: str, cost: Cost, duration: int):
        # print(f"short_form: {short_form}")
        # print(f"cost: {cost}")
        # print(f"duration: {duration}")
        self.short_form: str = short_form
        self.cost: Cost = Cost(cost)
        self.duration: int = duration
        # print(f"self.short_form: {self.short_form}")
        # print(f"self.cost: {self.cost}")
        # print(f"self.duration: {self.duration}")
        # sys.exit(1)

    def __repr__(self):
        return f"Data(short_form={self.short_form}, cost={self.cost}, duration={self.duration})"

    def graph(self) -> Graph:
        return from_graph6_bytes(self.short_form.encode('ascii'))
    
    @staticmethod
    def read_data(dir_name: str, batch_size: int) -> list[list[list['Data']]]:
        # Find all JSON files in the directory
        json_files: list[str] = glob.glob(os.path.join(dir_name, '*.json'))
        json_files.sort()
        data_lists: list[list['Data']] = []
        for file_path in json_files:
            # replace `(` and `)` with `[` and `]`
            with open(file_path, 'r') as f:
                data_list: list[list[dict]] = json.load(f)
                # print(type(data_list))
                # print(type(data_list[0]))
                # print(type(data_list[0][0]))
                # # any element
                # for _ in data_list:
                #     print(type(_))
                #     for __ in _:
                #         print(f"\t{type(__)}")
                #         for key, value in __.items():
                #             print(f"\t\t{key}: {value}")
                    # sys.exit(1)
                # sys.exit(1)
                # Convert dictionaries to Data objects
                if len(data_list) != batch_size:
                    print(f"Error: {file_path} has {len(data_list)} threads")
                    sys.exit(1)
                data_list: list = [[Data(**data_dict) for data_dict in sublist] for sublist in data_list]
                data_lists.append(data_list)
        return data_lists
