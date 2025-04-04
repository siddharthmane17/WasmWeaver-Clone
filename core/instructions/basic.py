from core.state.functions import Function
from core.state.state import GlobalState
from core.tile import AbstractTile
from core.value import I32, Num


class NoOp(AbstractTile):
    name = "NoOp"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        pass

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "nop"

    def get_byte_code_size(self):
        return 1


class Drop(AbstractTile):
    name = "Drop"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        return len(current_state.stack.get_current_frame().stack) > 0

    def apply(self, current_state: GlobalState, current_function: Function):
        current_state.stack.get_current_frame().stack_pop()

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "drop"

    def get_byte_code_size(self):
        return 1


class Select(AbstractTile):
    name = "Select"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        # Ensure there are at least three values on the stack
        if len(current_state.stack.get_current_frame().stack) < 3:
            return False
        #Check if the condition is an i32
        condition = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(condition, I32):
            return False

        if not isinstance(current_state.stack.get_current_frame().stack_peek(3), Num) or not isinstance(
                current_state.stack.get_current_frame().stack_peek(2), Num):
            return False
        #Check if the values are the same type
        #print(current_state.stack.get_current_frame().stack_peek(3), current_state.stack.get_current_frame().stack_peek(2))
        if type(current_state.stack.get_current_frame().stack_peek(3)) != type(
                current_state.stack.get_current_frame().stack_peek(2)):
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        condition = current_state.stack.get_current_frame().stack_pop()
        false_value = current_state.stack.get_current_frame().stack_pop()
        true_value = current_state.stack.get_current_frame().stack_pop()

        # Choose based on the condition
        # Non-zero condition selects the true value, zero selects the false value
        if condition.value != 0:
            result = true_value
        else:
            result = false_value

        # Push the result back onto the stack
        current_state.stack.get_current_frame().stack_push(result)

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "select"

    def get_byte_code_size(self):
        return 1
