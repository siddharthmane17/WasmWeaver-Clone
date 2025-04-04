from copy import deepcopy
from typing import List

from core.value import Val


class Global:
    """A simple global variable that stores a name and a value (with type information)."""
    def __init__(self,value: Val,name:str, mutable: bool):
        self.value = value
        self.mutable = mutable
        self.name = name
        self.init_value = deepcopy(value)

    def generate_code(self):
        if self.mutable:
            return f"(global ${self.name} (mut {self.value.get_wasm_type()}) ({self.init_value.to_init_str()}))"
        else:
            return f"(global ${self.name} {self.value.get_wasm_type()} ({self.init_value.to_init_str()}))"


class Globals:
    """A simple global state that stores global variables."""

    def __init__(self):
        self.globals : List[Global] = []

    def add(self, value: Global):
        """Adds a global variable to the state and returns the index."""
        self.globals.append(value)
        return len(self.globals) - 1

    def __getitem__(self, item):
        return self.globals[item]

    def get_global_by_name(self, name: str):
        """Returns the global variable with the given name."""
        for global_var in self.globals:
            if global_var.name == name:
                return global_var
        return None
