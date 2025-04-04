from typing import List, Type, Dict
from core.value import Val


class ExtFunction:
    def __init__(self,name: str, inputs: List[Type[Val]], outputs: List[Type[Val]], cpu_complexity, gpu_complexity):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.cpu_complexity = cpu_complexity
        self.gpu_complexity = gpu_complexity
        self.compute_callback = None

    def get_cpu_complexity(self, *args):
        print(args)
        return self.cpu_complexity(*args)

    def get_gpu_complexity(self, *args):
        return self.gpu_complexity(*args)

    def generate_code(self):
        return f'(import "env" "{self.name}" (func ${self.name} (param {" ".join([x.get_wasm_type() for x in self.inputs])}) (result {"".join([x.get_wasm_type() for x in self.outputs])})))'


class ExtFunctions:
    """A simple function state that stores functions."""

    def __init__(self):
        self.functions: Dict[str, ExtFunction] = {}

    def set(self, value: ExtFunction):
        self.functions[value.name] = value

    def get(self, name):
        return self.functions[name]

    def __len__(self):
        return len(self.functions)