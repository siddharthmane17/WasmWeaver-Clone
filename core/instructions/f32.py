
import random

import numpy as np

from core.config.config import MEMORY_MAX_WRITE_INDEX
from core.state.functions import Function
from core.state.state import GlobalState
from core.tile import AbstractTile
from core.value import F32, I32, F64, I64


class Float32Const(AbstractTile):
    name = "Float32Const"

    def __init__(self, seed: int):
        super().__init__(seed)
        # Choose an exponent for the floating point number
        exponent = random.randint(-126, 127)  # Avoid generating subnormal numbers
        # Generate a random float within the range of normal IEEE 754 floating-point numbers
        if random.randint(0, 50) == 0:
            # Special case for 0
            self.value = np.float32(0)
        else:
            self.value = np.float32(random.uniform(-2 ** exponent, 2 ** exponent))

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        return current_state.stack.get_current_frame().can_push_to_stack()

    def apply(self, current_state: GlobalState, current_function: Function):
        current_state.stack.get_current_frame().stack_push(F32(self.value))
        return current_state

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f"(f32.const {self.value})"

    def get_byte_code_size(self):
        # This is a simple model and might not directly correspond to actual byte code size
        # We assume all floats take the same size, but in a real implementation this could vary
        # based on the value itself
        return 5  # This is assuming constant size, adjust if a different model is used


class Float32Add(AbstractTile):
    name = "Float32Add"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(a.value + b.value))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.add"

    def get_byte_code_size(self):
        return 1


class Float32Sub(AbstractTile):
    name = "Float32Sub"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32((a.value - b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.sub"

    def get_byte_code_size(self):
        return 1


class Float32Mul(AbstractTile):
    name = "Float32Mul"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32((a.value * b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.mul"

    def get_byte_code_size(self):
        return 1


class Float32Div(AbstractTile):
    name = "Float32Div"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32) and b.value != 0  # ensure b is not zero

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        #print("Dividing", a.value, "by", b.value)
        if b.value == 0.0:
            raise ValueError("Division by zero")
        result = a.value.astype(np.float32) / b.value.astype(np.float32)
        result = np.float32(result)  # Round to nearest integer
        #print("Dividing", a.value, "by", b.value, "result is", result)
        current_state.stack.get_current_frame().stack_push(F32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.div"

    def get_byte_code_size(self):
        return 1

class Float32Sqrt(AbstractTile):
    name = "Float32Sqrt"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and a.value >= 0

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        # print("Sqrt", a.value)
        # print(np.sqrt(a.value))
        current_state.stack.get_current_frame().stack_push(F32(np.sqrt(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.sqrt"

    def get_byte_code_size(self):
        return 1

class Float32Min(AbstractTile):
    name = "Float32Min"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.minimum(a.value, b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.min"

    def get_byte_code_size(self):
        return 1

class Float32Max(AbstractTile):
    name = "Float32Max"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.maximum(a.value, b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.max"

    def get_byte_code_size(self):
        return 1

class Float32Ceil(AbstractTile):
    name = "Float32Ceil"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.ceil(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.ceil"

    def get_byte_code_size(self):
        return 1


class Float32Floor(AbstractTile):
    name = "Float32Floor"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.floor(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.floor"

    def get_byte_code_size(self):
        return 1

class Float32Trunc(AbstractTile):
    name = "Float32Trunc"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.trunc(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.trunc"

    def get_byte_code_size(self):
        return 1

class Float32Nearest(AbstractTile):
    name = "Float32Nearest"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.round(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.nearest"

    def get_byte_code_size(self):
        return 1

class Float32Abs(AbstractTile):
    name = "Float32Abs"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.abs(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.abs"

    def get_byte_code_size(self):
        return 1

class Float32Neg(AbstractTile):
    name = "Float32Neg"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(-a.value))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.neg"

    def get_byte_code_size(self):
        return 1

class Float32CopySign(AbstractTile):
    name = "Float32CopySign"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.copysign(a.value, b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.copysign"

    def get_byte_code_size(self):
        return 1


class Float32Eq(AbstractTile):
    name = "Float32Eq"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value == b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.eq"

    def get_byte_code_size(self):
        return 1

class Float32Ne(AbstractTile):
    name = "Float32Ne"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value != b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.ne"

    def get_byte_code_size(self):
        return 1

class Float32Lt(AbstractTile):
    name = "Float32Lt"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value < b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.lt"

    def get_byte_code_size(self):
        return 1

class Float32Le(AbstractTile):
    name = "Float32Le"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value <= b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.le"

    def get_byte_code_size(self):
        return 1

class Float32Gt(AbstractTile):
    name = "Float32Gt"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value > b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.gt"

    def get_byte_code_size(self):
        return 1


class Float32Ge(AbstractTile):
    name = "Float32Ge"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32) and isinstance(b, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = np.int32(a.value >= b.value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.ge"

    def get_byte_code_size(self):
        return 1

class Float32DemoteF64(AbstractTile):
    name = "Float32DemoteF64"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F64)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.float32(a.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.demote_f64"

    def get_byte_code_size(self):
        return 1

class Float32ConvertI32S(AbstractTile):
    name = "Float32ConvertI32S"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.float32(np.int32(a.value))))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.convert_i32_s"

    def get_byte_code_size(self):
        return 1

class Float32ConvertI32U(AbstractTile):
    name = "Float32ConvertI32U"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.float32(np.uint32(a.value))))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.convert_i32_u"

    def get_byte_code_size(self):
        return 1

class Float32ConvertI64S(AbstractTile):
    name = "Float32ConvertI64S"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I64)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.float32(np.int64(a.value))))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.convert_i64_s"

    def get_byte_code_size(self):
        return 1

class Float32ConvertI64U(AbstractTile):
    name = "Float32ConvertI64U"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I64)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(np.float32(np.uint64(a.value))))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.convert_i64_u"

    def get_byte_code_size(self):
        return 1

class Float32ReinterpretI32(AbstractTile):
    name = "Float32ReinterpretI32"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        result = a.view(np.float32)
        current_state.stack.get_current_frame().stack_push(F32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "f32.reinterpret_i32"

    def get_byte_code_size(self):
        return 1


class Float32Store(AbstractTile):
    name = "Float32Store"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(2)
        if not isinstance(offset, I32):
            return False
        value = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(value, F32):
            return False
        #Check if in range
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 4:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop()
        offset = current_state.stack.get_current_frame().stack_pop()
        current_state.memory.f32_store(offset.value.astype(np.uint32), value.value.astype(np.float32))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f"f32.store"

    def get_byte_code_size(self):
        return 2


class Float32Load(AbstractTile):
    name = "Float32Load"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(offset, I32):
            return False
        #Check if in range
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 4:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        offset = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(F32(current_state.memory.f32_load(offset.value.astype(np.uint32))))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f"f32.load"

    def get_byte_code_size(self):
        return 2
