import numpy as np

from core.state.functions import Function
from core.state.state import GlobalState
from core.tile import AbstractTile
from core.value import I32, I64, F32, F64


class Canary(AbstractTile):
    name = "Canary"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        n = current_state.stack.get_current_frame().stack_peek()
        print("Canary:", n)
        if isinstance(n, I32):
            current_state.canary_output.append(int(np.int32(n.value)))
        elif isinstance(n, I64):
            current_state.canary_output.append(int(np.int64(n.value)))
        elif isinstance(n, F32):
            current_state.canary_output.append(float(np.float32(n.value)))
        elif isinstance(n, F64):
            current_state.canary_output.append(float(np.float64(n.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return ""

    def get_byte_code_size(self):
        return 0

    def get_fuel_cost(self):
        return 0

    def get_response_time(self):
        return 0