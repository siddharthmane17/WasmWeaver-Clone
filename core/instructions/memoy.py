import numpy as np

from core.state.functions import Function
from core.state.state import GlobalState
from core.tile import AbstractTile
from core.value import I32


#class MemorySize(AbstractTile):
#    name = "Memory size"

#    def __init__(self, seed: int):
#        super().__init__(seed)

#    @staticmethod
#    def can_be_placed(current_state: GlobalState, current_function: Function):
#        return current_state.stack.get_current_frame().can_push_to_stack()

#    def apply(self, current_state: GlobalState, current_function: Function):
        #In our case, the memory size is always 1
#        current_state.stack.get_current_frame().stack_push(I32(np.int32(1)))

#    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
#        return f"memory.size"

#    def get_byte_code_size(self):
#        return 1
