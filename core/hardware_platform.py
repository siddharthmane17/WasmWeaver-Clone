import json
import os
from typing import Literal

import torch


class HardwarePlatform:

    def __init__(self, platform_type: Literal["CPU","GPU"]):
        self.platform_type = platform_type
        self.response_times = {}
        if platform_type == "CPU":
            print("Loading CPU response times")
            # Get all json files in the cpu folder
            fs = os.listdir("dataset/validation/CPU")
            for f in fs:
                print("Loading file", f)
                with open(f"dataset/validation/CPU/{f}") as json_file:
                    data = json.load(json_file)
                    #print(data)
                    for seed, resp in data:
                        self.response_times[seed] = resp
        elif platform_type == "GPU":
            print("Loading GPU response times")
            # Get all json files in the gpu folder
            fs = os.listdir("dataset/validation/GPU")
            for f in fs:
                print("Loading file", f)
                with open(f"dataset/validation/GPU/{f}") as json_file:
                    data = json.load(json_file)
                    for seed, resp in data:
                        self.response_times[seed] = resp

    def get_response_time_by_seed(self, seed: int):
        seed = int(seed)
        if seed not in self.response_times:
            raise ValueError(f"Seed {seed} not found in response times")
        return self.response_times[seed]

    def torch_response_time_by_seeds(self, seeds:torch.tensor):
        return torch.log(torch.tensor([self.get_response_time_by_seed(seed) for seed in seeds]))+30


if __name__ == "__main__":
    cpu_platform = HardwarePlatform("CPU")
    print(cpu_platform.get_response_time_by_seed(4000000075233))
    gpu_platform = HardwarePlatform("GPU")
    print(gpu_platform.get_response_time_by_seed(4000000075233))