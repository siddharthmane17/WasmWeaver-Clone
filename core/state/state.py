import copy
from typing import TypeVar

from core.constraints import Constraints
from core.state.functions import Functions
from core.state.functions_ext import ExtFunctions
from core.state.globals import Globals
from core.state.memory import Memory
from core.state.stack import Stack
from core.state.tables import Tables

T = TypeVar('T')


class GlobalState:
    """Combines multiple states like stack, memory, and locals into one global state."""
    def __init__(self):
        self.checkpoints = {}
        self.globals = Globals()
        self.tables = Tables()
        self.functions = Functions()
        self.memory = Memory()
        self.stack = Stack()
        self.constraints = Constraints()
        self.ext_functions = ExtFunctions()
        self.canary_output = []

    def create_checkpoint(self) -> int:
        """Creates a checkpoint of the current state and returns the id of the checkpoint."""
        i = 0
        while i in self.checkpoints:
            i += 1
        self.checkpoints[i] = copy.deepcopy({"globals": self.globals,
                                             "functions": self.functions,
                                             "memory": self.memory,
                                             "tables": self.tables,
                                             "stack": self.stack,
                                             "ext_functions": self.ext_functions,
                                             "constraints": self.constraints})
        return i

    def restore_checkpoint(self, i: int, delete=False):
        """Restores the state to the state of the checkpoint with the given id."""
        self.globals = copy.deepcopy(self.checkpoints[i]["globals"])
        self.functions = copy.deepcopy(self.checkpoints[i]["functions"])
        self.memory = copy.deepcopy(self.checkpoints[i]["memory"])
        self.tables = copy.deepcopy(self.checkpoints[i]["tables"])
        self.stack = copy.deepcopy(self.checkpoints[i]["stack"])
        self.ext_functions = copy.deepcopy(self.checkpoints[i]["ext_functions"])
        self.constraints = copy.deepcopy(self.checkpoints[i]["constraints"])
        if delete:
            del self.checkpoints[i]

    def delete_checkpoint(self, i: int):
        """Deletes the checkpoint with the given id."""
        del self.checkpoints[i]

