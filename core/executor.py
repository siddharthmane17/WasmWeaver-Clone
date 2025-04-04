import numpy as np


class AbstractPlatform:

    def __init__(self, name: str, cpu_cycles_per_second: float, gpu_cycles_per_second: float = -1):
        self.name = name
        self.gpu_cycles_per_second = gpu_cycles_per_second
        self.cpu_cycles_per_second = cpu_cycles_per_second

    def calculate_response_time(self, abstract_run_result: "AbstractRunResult") -> float:
        cycles_per_instruction_average = 10
        total_time = cycles_per_instruction_average * abstract_run_result.fuel / self.cpu_cycles_per_second
        for ext_res in abstract_run_result.ext_resources:
            if self.gpu_cycles_per_second == -1:
                total_time += ext_res.cpu_cycles / self.cpu_cycles_per_second
            else:
                # If gpu cycles smaller then cpu cycles, use cpu cycles / 50 as a rough estimate
                if ext_res.gpu_cycles < ext_res.cpu_cycles:
                    total_time += (ext_res.cpu_cycles/50) / self.cpu_cycles_per_second
                else:

                    total_time += ext_res.cpu_cycles / self.cpu_cycles_per_second
        return total_time

    def response_time_to_fuel(self, response_time: np.array) -> np.array:
        return response_time * self.cpu_cycles_per_second / 10
