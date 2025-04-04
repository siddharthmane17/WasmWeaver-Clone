
from core.config.config import MEMORY_MAX_WRITE_INDEX
from core.state.functions import Function
from core.state.state import GlobalState
from core.tile import AbstractTile
from core.value import I32, I64, F32, F64
import random
import math
import numpy as np


class Int32Const(AbstractTile):
    name = "Int32Const"

    def __init__(self, seed: int):
        super().__init__(seed)
        # Choose an order of magnitude such that 10**magnitude is within a 32-bit integer range
        magnitude = 10 ** random.randint(0, 9)  # Choose magnitudes from 10^0 to 10^9
        # Randomly choose a number within this magnitude, adjusting for both positive and negative ranges
        if random.randint(0,50) == 0:
            #Special case for 0
            self.value = np.int32(0)
        elif magnitude == 1:
            self.value = np.int32(random.randint(-2 ** 31, 2 ** 31 - 1))  # Full range for smallest magnitude
        else:
            self.value = np.int32(random.randint(-magnitude, magnitude - 1))

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        return current_state.stack.get_current_frame().can_push_to_stack()

    def apply(self, current_state: GlobalState, current_function: Function):
        current_state.stack.get_current_frame().stack_push(I32(self.value))
        return current_state

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f"(i32.const {self.value})"

    def get_byte_code_size(self):
        if -64 <= self.value < 64:
            return 2
        elif -8192 <= self.value < 8191:
            return 3
        elif -1048576 <= self.value < 1048575:
            return 4
        else:
            return 5


class Int32Add(AbstractTile):
    name = "Int32Add"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(I32(a.value + b.value))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.add"

    def get_byte_code_size(self):
        return 1


class Int32Sub(AbstractTile):
    name = "Int32Sub"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(I32((a.value - b.value)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.sub"

    def get_byte_code_size(self):
        return 1


class Int32Mul(AbstractTile):
    name = "Int32Mul"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        a = current_state.stack.get_current_frame().stack_pop()
        b = current_state.stack.get_current_frame().stack_pop()
        current_state.stack.get_current_frame().stack_push(I32((a.value * b.value) ))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.mul"

    def get_byte_code_size(self):
        return 1


class Int32DivS(AbstractTile):
    name = "Int32DivS"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32) and b.value != 0  # ensure b is not zero

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        if b.value == 0:
            raise ValueError("Division by zero")
        result = a.value.astype(np.int32) / b.value.astype(np.int32)
        result = np.int32(result)  # Round to nearest integer
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.div_s"

    def get_byte_code_size(self):
        return 1


class Int32DivU(AbstractTile):
    name = "Int32DivU"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32) and b.value != 0  # ensure b is not zero for division

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        if b.value == 0:
            raise ValueError("Division by zero")
        # Ensure the operands are treated as unsigned by using modulo 2**32
        result = (a.value.astype(np.uint32) // b.value.astype(np.uint32))
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.div_u"

    def get_byte_code_size(self):
        return 1


class Int32RemS(AbstractTile):
    name = "Int32RemS"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32) and b.value != 0  # ensure b is not zero for rem

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        if b.value == 0:
            raise ValueError("Division by zero")
        result = a.value.astype(np.int32) % b.value.astype(np.int32)
        # Adjust result to make sure the sign matches the dividend's sign
        if (a.value.astype(np.int32) < 0) != (b.value.astype(np.int32) < 0) and result != 0:
            result = result - b.value.astype(np.int32)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.rem_s"

    def get_byte_code_size(self):
        return 1


class Int32RemU(AbstractTile):
    name = "Int32RemU"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32) and b.value != 0  # ensure b is not zero

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        if b.value == 0:
            raise ValueError("Division by zero")
        # Calculate the remainder treating values as unsigned
        result = ((a.value.astype(np.uint32)) % (b.value.astype(np.uint32)))
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.rem_u"

    def get_byte_code_size(self):
        return 1


class Int32And(AbstractTile):
    name = "Int32And"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        # Ensure there are at least two values on the stack
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        # Check if both are I32 instances
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        # Pop the top two values from the stack
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        # Apply the bitwise AND operation
        result = a.value & b.value
        # Push the result back to the stack
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.and"

    def get_byte_code_size(self):
        return 1


class Int32Or(AbstractTile):
    name = "Int32Or"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = a.value | b.value  # Bitwise OR operation
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.or"

    def get_byte_code_size(self):
        return 1


class Int32Xor(AbstractTile):
    name = "Int32Xor"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop()
        a = current_state.stack.get_current_frame().stack_pop()
        result = a.value ^ b.value  # Bitwise XOR operation
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.xor"

    def get_byte_code_size(self):
        return 1


class Int32Shl(AbstractTile):
    name = "Int32Shl"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        shift_amount = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32) % 32
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        result = (value << shift_amount)  # Ensure 32-bit wrapping
        current_state.stack.get_current_frame().stack_push(I32(result))
        #print("Shifting", value, "left by", shift_amount, "result is", result)

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.shl"

    def get_byte_code_size(self):
        return 1


class Int32ShrS(AbstractTile):
    name = "Int32ShrS"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        shift_amount = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32) % 32
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        result = value >> shift_amount  # Python's `>>` performs an arithmetic shift for signed integers
        current_state.stack.get_current_frame().stack_push(I32(result.astype(np.int32)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.shr_s"

    def get_byte_code_size(self):
        return 1


class Int32ShrU(AbstractTile):
    name = "Int32ShrU"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        shift_amount = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32) % 32
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        result = value >> shift_amount  # Unsigned shift
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.shr_u"

    def get_byte_code_size(self):
        return 1


class Int32Rotl(AbstractTile):
    name = "Int32Rotl"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        rotate_amount = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32) % 32
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Perform rotate left
        result = value & 0xFFFFFFFF  # Ensure 32-bit wrapping
        for i in range(rotate_amount):
            result = (result << 1) | ((result & 0x80000000) >> 31)
            result = result & 0xFFFFFFFF

        #print("Rotating", value, "left by", rotate_amount, "result is", result)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.rotl"

    def get_byte_code_size(self):
        return 1


class Int32Rotr(AbstractTile):
    name = "Int32Rotr"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        rotate_amount = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32) % 32
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Perform rotate right
        result = value & 0xFFFFFFFF  # Ensure 32-bit wrapping
        for i in range(rotate_amount):
            result = (result >> 1) | ((result & 1) << 31)
            result = result & 0xFFFFFFFF  # Ensure 32-bit wrapping

        #print("Rotating", value, "right by", rotate_amount, "result is", result)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.rotr"

    def get_byte_code_size(self):
        return 1


class Int32Clz(AbstractTile):
    name = "Int32Clz"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        leading_zeros = 0
        for i in range(32):  # Iterate over each bit in a 32-bit integer
            if (value & (1 << (31 - i))) != 0:
                break
            leading_zeros += 1 # Use log2 to find the position of the highest set bit
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(leading_zeros)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.clz"

    def get_byte_code_size(self):
        return 1


class Int32Ctz(AbstractTile):
    name = "Int32Ctz"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        if value == 0:
            result = 32  # If the value is zero, all bits are zero, hence 32 trailing zeros
        else:
            result = 0
            while (value & 1) == 0:
                result += 1
                value >>= 1  # Right shift the value until a 1 is found
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.ctz"

    def get_byte_code_size(self):
        return 1


class Int32Popcnt(AbstractTile):
    name = "Int32Popcnt"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Count the number of 1s in the binary representation of the value
        result = bin(value & 0xFFFFFFFF).count('1')
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.popcnt"

    def get_byte_code_size(self):
        return 1


class Int32Eqz(AbstractTile):
    name = "Int32Eqz"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Check if the value is zero and return 1 if true, else 0
        result = 1 if value == 0 else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.eqz"

    def get_byte_code_size(self):
        return 1


class Int32Eq(AbstractTile):
    name = "Int32Eq"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Check if the two values are equal and return 1 if true, else 0
        result = 1 if a == b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.eq"

    def get_byte_code_size(self):
        return 1


class Int32Ne(AbstractTile):
    name = "Int32Ne"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        a = current_state.stack.get_current_frame().stack_peek(2)
        b = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Check if the two values are not equal and return 1 if true, else 0
        result = 1 if a != b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.ne"

    def get_byte_code_size(self):
        return 1


class Int32LtS(AbstractTile):
    name = "Int32LtS"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        b = current_state.stack.get_current_frame().stack_peek(1)
        a = current_state.stack.get_current_frame().stack_peek(2)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        # Check if a is less than b and return 1 if true, else 0
        result = 1 if a < b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.lt_s"

    def get_byte_code_size(self):
        return 1


class Int32LtU(AbstractTile):
    name = "Int32LtU"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        b = current_state.stack.get_current_frame().stack_peek(1)
        a = current_state.stack.get_current_frame().stack_peek(2)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Check if a is less than b and return 1 if true, else 0
        result = 1 if a < b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.lt_u"

    def get_byte_code_size(self):
        return 1


class Int32LeS(AbstractTile):
    name = "Int32LeS"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        b = current_state.stack.get_current_frame().stack_peek(1)
        a = current_state.stack.get_current_frame().stack_peek(2)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        # Check if a is less than or equal to b and return 1 if true, else 0
        result = 1 if a <= b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.le_s"

    def get_byte_code_size(self):
        return 1


class Int32LeU(AbstractTile):
    name = "Int32LeU"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        b = current_state.stack.get_current_frame().stack_peek(1)
        a = current_state.stack.get_current_frame().stack_peek(2)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Check if a is less than or equal to b and return 1 if true, else 0
        result = 1 if a <= b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.le_u"

    def get_byte_code_size(self):
        return 1


class Int32GtS(AbstractTile):
    name = "Int32GtS"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        b = current_state.stack.get_current_frame().stack_peek(1)
        a = current_state.stack.get_current_frame().stack_peek(2)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        # Check if a is greater than b and return 1 if true, else 0
        result = 1 if a > b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.gt_s"

    def get_byte_code_size(self):
        return 1


class Int32GtU(AbstractTile):
    name = "Int32GtU"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        b = current_state.stack.get_current_frame().stack_peek(1)
        a = current_state.stack.get_current_frame().stack_peek(2)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Check if a is greater than b and return 1 if true, else 0
        result = 1 if a > b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.gt_u"

    def get_byte_code_size(self):
        return 1


class Int32GeS(AbstractTile):
    name = "Int32GeS"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        b = current_state.stack.get_current_frame().stack_peek(1)
        a = current_state.stack.get_current_frame().stack_peek(2)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        # Check if a is greater than or equal to b and return 1 if true, else 0
        result = 1 if a >= b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.ge_s"

    def get_byte_code_size(self):
        return 1


class Int32GeU(AbstractTile):
    name = "Int32GeU"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        b = current_state.stack.get_current_frame().stack_peek(1)
        a = current_state.stack.get_current_frame().stack_peek(2)
        return isinstance(a, I32) and isinstance(b, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        b = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        a = current_state.stack.get_current_frame().stack_pop().value.astype(np.uint32)
        # Check if a is greater than or equal to b and return 1 if true, else 0
        result = 1 if a >= b else 0
        current_state.stack.get_current_frame().stack_push(I32(np.uint32(result)))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.ge_u"

    def get_byte_code_size(self):
        return 1


class Int32WrapI64(AbstractTile):
    name = "Int32WrapI64"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I64)

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.int64)
        # Wrap the 64-bit integer to a 32-bit integer with modulo 2**32
        result = np.int32(value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.wrap_i64"

    def get_byte_code_size(self):
        return 1


class Int32TruncF32S(AbstractTile):
    name = "Int32TruncF32S"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(a, F32):
            return False
        if a.value < -2**31 or a.value >= 2**31:
            return False
        if math.isnan(a.value) or math.isinf(a.value):
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.float32)
        # Truncate the 32-bit float to a 32-bit integer
        result = np.int32(value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.trunc_f32_s"

    def get_byte_code_size(self):
        return 1


class Int32TruncF64S(AbstractTile):
    name = "Int32TruncF64S"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(a, F64):
            return False
        if a.value < -2**31 or a.value >= 2**31:
            return False
        if math.isnan(a.value) or math.isinf(a.value):
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.float64)
        # Truncate the 64-bit float to a 32-bit integer
        result = np.int32(value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.trunc_f64_s"

    def get_byte_code_size(self):
        return 1


class Int32TruncF32U(AbstractTile):
    name = "Int32TruncF32U"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(a, F32):
            return False
        if a.value < 0 or a.value >= 2**32:
            return False
        if math.isnan(a.value) or math.isinf(a.value):
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.float32)
        # Truncate the 32-bit float to a 32-bit unsigned integer
        result = np.uint32(value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.trunc_f32_u"

    def get_byte_code_size(self):
        return 1


class Int32TruncF64U(AbstractTile):
    name = "Int32TruncF64U"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(a, F64):
            return False
        if a.value < 0 or a.value >= 2**32:
            return False
        if math.isnan(a.value) or math.isinf(a.value):
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.float64)
        # Truncate the 64-bit float to a 32-bit unsigned integer
        # print("Truncating", value)
        # print("Truncated to", np.uint32(value))
        result = np.uint32(value)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.trunc_f64_u"

    def get_byte_code_size(self):
        return 1


class Int32ReinterpretF32(AbstractTile):

    name = "Int32ReinterpretF32"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, F32)

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.float32)
        # Reinterpret the 32-bit float as a 32-bit integer with 1:1 bit pattern
        result = value.view(np.int32)
        #print("Reinterpreting", value, "as", result)
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.reinterpret_f32"

    def get_byte_code_size(self):
        return 1


class Int32Extend8S(AbstractTile):
    name = "Int32Extend8S"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        # Sign-extend the 8-bit integer to a 32-bit integer
        result = np.int32(np.int8(value))
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.extend8_s"

    def get_byte_code_size(self):
        return 1


class Int32Extend16S(AbstractTile):
    name = "Int32Extend16S"

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        a = current_state.stack.get_current_frame().stack_peek(1)
        return isinstance(a, I32)

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop().value.astype(np.int32)
        # Sign-extend the 16-bit integer to a 32-bit integer
        result = np.int32(np.int16(value))
        current_state.stack.get_current_frame().stack_push(I32(result))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.extend16_s"

    def get_byte_code_size(self):
        return 1


class Int32Store(AbstractTile):
    name = "Int32Store"

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
        if not isinstance(value, I32):
            return False
        #Check if in range
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 4:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop()
        offset = current_state.stack.get_current_frame().stack_pop()
        current_state.memory.i32_store(offset.value.astype(np.uint32), value.value.astype(np.uint32))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f"i32.store"

    def get_byte_code_size(self):
        return 2



class Int32Store8(AbstractTile):
    name = "Int32Store8"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(2)
        value = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(offset, I32) or not isinstance(value, I32):
            return False
        # Check range for 8-bit store
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 1:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop()
        offset = current_state.stack.get_current_frame().stack_pop()
        # Store only the least significant 8 bits of the integer
        current_state.memory.i32_store8(offset.value.astype(np.uint32), value.value.astype(np.uint32) & 0xFF)

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f"i32.store8"

    def get_byte_code_size(self):
        return 2



class Int32Store16(AbstractTile):
    name = "Int32Store16"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 2:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(2)
        value = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(offset, I32) or not isinstance(value, I32):
            return False
        # Check range for 16-bit store
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 2:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        value = current_state.stack.get_current_frame().stack_pop()
        offset = current_state.stack.get_current_frame().stack_pop()
        # Store only the least significant 16 bits of the integer
        current_state.memory.i32_store16(offset.value.astype(np.uint32), value.value.astype(np.uint32) & 0xFFFF)

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f"i32.store16"

    def get_byte_code_size(self):
        return 2



class Int32Load(AbstractTile):
    name = "Int32Load"

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
        value = current_state.memory.i32_load(offset.value.astype(np.uint32))
        current_state.stack.get_current_frame().stack_push(I32(value))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f"i32.load"

    def get_byte_code_size(self):
        return 2



class Int32Load8U(AbstractTile):
    name = "Int32Load8U"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(offset, I32):
            return False
        # Check if in range for 8-bit load
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 1:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        offset = current_state.stack.get_current_frame().stack_pop()
        value = current_state.memory.i32_load8_u(offset.value.astype(np.uint32))
        current_state.stack.get_current_frame().stack_push(I32(value))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.load8_u"

    def get_byte_code_size(self):
        return 2


class Int32Load8S(AbstractTile):
    name = "Int32Load8S"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(offset, I32):
            return False
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 1:
            return False
        #print(offset.value)
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        offset = current_state.stack.get_current_frame().stack_pop()
        value = current_state.memory.i32_load8_s(offset.value.astype(np.uint32))
        current_state.stack.get_current_frame().stack_push(I32(value))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.load8_s"

    def get_byte_code_size(self):
        return 2



class Int32Load16U(AbstractTile):
    name = "Int32Load16U"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(1)
        if not isinstance(offset, I32):
            return False
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 2:
            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):
        offset = current_state.stack.get_current_frame().stack_pop()
        value = current_state.memory.i32_load16_u(offset.value.astype(np.uint32))
        current_state.stack.get_current_frame().stack_push(I32(value))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.load16_u"

    def get_byte_code_size(self):
        return 2


class Int32Load16S(AbstractTile):
    name = "Int32Load16S"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        if len(current_state.stack.get_current_frame().stack) < 1:
            return False
        offset = current_state.stack.get_current_frame().stack_peek(1)

        if not isinstance(offset, I32):
            return False
        if offset.value.astype(np.uint32) >= MEMORY_MAX_WRITE_INDEX - 2:

            return False
        return True

    def apply(self, current_state: GlobalState, current_function: Function):

        offset = current_state.stack.get_current_frame().stack_pop()
        #print("t: ", offset.value.astype(np.uint32))
        value = current_state.memory.i32_load16_s(offset.value.astype(np.uint32))
        current_state.stack.get_current_frame().stack_push(I32(value))

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return "i32.load16_s"

    def get_byte_code_size(self):
        return 2
