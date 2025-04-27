import random
from typing import List

import numpy as np

from core.config.config import MEMORY_MAX_WRITE_INDEX
from core.state.functions import Function, Block
from core.state.state import GlobalState
from core.tile import AbstractTile
from core.value import F32, I32, F64, I64

class Float64Const(AbstractTile):
    name = "Float64Const"

    def __init__(self, seed: int):
        super().__init__(seed)
        # Choose an exponent for the floating point number

        # Generate a random float within the range of normal IEEE 754 floating-point numbers
        if random.randint(0, 50) == 0:
            # Special case for 0
            self.value = np.float64(0)
        else:
            exponent = random.randint(-1022, 1023)  # Avoid generating subnormal numbers
            significand = random.uniform(1, 2)
            self.value = np.float64(significand * 2 ** exponent)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        return current_state.stack.get_current_frame().can_push_to_stack()

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        current_state.stack.get_current_frame().stack_push(F64(self.value))
        return current_state

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return f"(f64.const {self.value})"

    def get_byte_code_size(self):
        # This is a simple model and might not directly correspond to actual byte code size
        # We assume all floats take the same size, but in a real implementation this could vary
        # based on the value itself
        return 9  # This is assuming constant size, adjust if a different model is used


class Float64Add(AbstractTile):
    name = "Float64Add"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(a.value + b.value))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.add"

    def get_byte_code_size(self):
        return 1


class Float64Sub(AbstractTile):
    name = "Float64Sub"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64((a.value - b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.sub"

    def get_byte_code_size(self):
        return 1


class Float64Mul(AbstractTile):
    name = "Float64Mul"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64((a.value * b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.mul"

    def get_byte_code_size(self):
        return 1


class Float64Div(AbstractTile):
    name = "Float64Div"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64) and b.value != 0  # ensure b is not zero

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        #print("Dividing", a.value, "by", b.value)
        if b.value == 0.0:
            raise ValueError("Division by zero")
        result = a.value.astype(np.float64) / b.value.astype(np.float64)
        result = np.float64(result)  # Round to nearest integer
        #print("Dividing", a.value, "by", b.value, "result is", result)
        current_state.stack.get_current_frame().stack_push(F64(result))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.div"

    def get_byte_code_size(self):
        return 1

class Float64Sqrt(AbstractTile):
    name = "Float64Sqrt"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and a.value >= 0

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        # print("Sqrt64 of", a.value)
        current_state.stack.get_current_frame().stack_push(F64(np.sqrt(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.sqrt"

    def get_byte_code_size(self):
        return 1

class Float64Min(AbstractTile):
    name = "Float64Min"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.minimum(a.value, b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.min"

    def get_byte_code_size(self):
        return 1

class Float64Max(AbstractTile):
    name = "Float64Max"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.maximum(a.value, b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.max"

    def get_byte_code_size(self):
        return 1

class Float64Ceil(AbstractTile):
    name = "Float64Ceil"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.ceil(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.ceil"

    def get_byte_code_size(self):
        return 1


class Float64Floor(AbstractTile):
    name = "Float64Floor"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        # print("Flooring", a.value)
        # print(np.floor(a.value))
        current_state.stack.get_current_frame().stack_push(F64(np.floor(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.floor"

    def get_byte_code_size(self):
        return 1

class Float64Trunc(AbstractTile):
    name = "Float64Trunc"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.trunc(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.trunc"

    def get_byte_code_size(self):
        return 1

class Float64Nearest(AbstractTile):
    name = "Float64Nearest"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.round(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.nearest"

    def get_byte_code_size(self):
        return 1

class Float64Abs(AbstractTile):
    name = "Float64Abs"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.abs(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.abs"

    def get_byte_code_size(self):
        return 1

class Float64Neg(AbstractTile):
    name = "Float64Neg"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(-a.value))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.neg"

    def get_byte_code_size(self):
        return 1

class Float64CopySign(AbstractTile):
    name = "Float64CopySign"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.copysign(a.value, b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.copysign"

    def get_byte_code_size(self):
        return 1


class Float64Eq(AbstractTile):
    name = "Float64Eq"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value == b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.eq"

    def get_byte_code_size(self):
        return 1

class Float64Ne(AbstractTile):
    name = "Float64Ne"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value != b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.ne"

    def get_byte_code_size(self):
        return 1

class Float64Lt(AbstractTile):
    name = "Float64Lt"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value < b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.lt"

    def get_byte_code_size(self):
        return 1

class Float64Le(AbstractTile):
    name = "Float64Le"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value <= b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.le"

    def get_byte_code_size(self):
        return 1

class Float64Gt(AbstractTile):
    name = "Float64Gt"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value > b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.gt"

    def get_byte_code_size(self):
        return 1


class Float64Ge(AbstractTile):
    name = "Float64Ge"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64) and isinstance(b, F64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value >= b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.ge"

    def get_byte_code_size(self):
        return 1

class Float64PromoteF32(AbstractTile):
    name = "Float64PromoteF32"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        # print("Promoting", a.value, "to float64")
        # print(np.float64(a.value))
        current_state.stack.get_current_frame().stack_push(F64(np.float64(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.promote_f32"

    def get_byte_code_size(self):
        return 1

class Float64ConvertI32S(AbstractTile):
    name = "Float64ConvertI32S"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.float64(np.int32(a.value))))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.convert_i32_s"

    def get_byte_code_size(self):
        return 1

class Float64ConvertI32U(AbstractTile):
    name = "Float64ConvertI32U"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.float64(np.uint32(a.value))))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.convert_i32_u"

    def get_byte_code_size(self):
        return 1

class Float64ConvertI64S(AbstractTile):
    name = "Float64ConvertI64S"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.float64(np.int64(a.value))))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.convert_i64_s"

    def get_byte_code_size(self):
        return 1

class Float64ConvertI64U(AbstractTile):
    name = "Float64ConvertI64U"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(np.float64(np.uint64(a.value))))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.convert_i64_u"

    def get_byte_code_size(self):
        return 1

class Float64ReinterpretI64(AbstractTile):
    name = "Float64ReinterpretI64"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I64)

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.int64)
        result = a.view(np.float64)
        current_state.stack.get_current_frame().stack_push(F64(result))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return "f64.reinterpret_i64"

    def get_byte_code_size(self):
        return 1


class Float64Store(AbstractTile):
    name = "Float64Store"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(2)
        if not isinstance(offset, I32):
            return False
        value = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(value, F64):
            return False
        #Check if in range
        if offset.value.astype(np.uint64) >= MEMORY_MAX_WRITE_INDEX - 8:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        value = current_state.stack.get_current_frame().stack_pop()
        offset = current_state.stack.get_current_frame().stack_pop()
        current_state.memory.f64_store(offset.value.astype(np.uint32), value.value.astype(np.float64))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return f"f64.store"

    def get_byte_code_size(self):
        return 2



class Float64Load(AbstractTile):
    name = "Float64Load"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(offset, I32):
            return False
        #Check if in range
        if offset.value.astype(np.uint64) >= MEMORY_MAX_WRITE_INDEX - 8:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        offset = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F64(current_state.memory.f64_load(offset.value.astype(np.uint32))))

    def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
        return f"f64.load"

    def get_byte_code_size(self):
        return 2

