import random
from typing import TYPE_CHECKING, Union

import numpy as np
import wasmtime

if TYPE_CHECKING:
    from core.state.functions import Function


class Val:

    def __init__(self, value):
        self.value = value

    @staticmethod
    def get_wasm_type():
        raise NotImplementedError

    @staticmethod
    def get_random_val():
        raise NotImplementedError

    @staticmethod
    def get_wasmtime_type():
        raise NotImplementedError

    @staticmethod
    def get_default_value():
        raise NotImplementedError

    def to_init_str(self):
        raise NotImplementedError

    def __str__(self):
        return "Val: " + str(self.value)


#Number types
class Num(Val):

    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        return "Num: " + str(self.value)


class I32(Num):

    def __init__(self, value=np.int32(0)):
        super().__init__(value)

    @staticmethod
    def get_wasm_type():
        return "i32"

    @staticmethod
    def get_random_val():
        return I32(np.int32(random.randint(-2 ** 31, 2 ** 31 - 1)))

    @staticmethod
    def get_wasmtime_type():
        return wasmtime.ValType.i32()

    def to_init_str(self):
        return f"i32.const {self.value}"

    @staticmethod
    def get_default_value():
        return I32(np.int32(0))

    def __str__(self):
        return "I32: " + str(self.value)


class I64(Num):

    def __init__(self, value=np.int64(0)):
        super().__init__(value)

    @staticmethod
    def get_wasm_type():
        return "i64"

    @staticmethod
    def get_wasmtime_type():
        return wasmtime.ValType.i64()

    @staticmethod
    def get_random_val():
        return I64(np.int64(random.randint(-2 ** 63, 2 ** 63 - 1)))

    @staticmethod
    def get_default_value():
        return I64(np.int64(0))

    def to_init_str(self):
        return f"i64.const {self.value}"

    def __str__(self):
        return "I64: " + str(self.value)

class F32(Num):

    def __init__(self, value=np.float32(0.0)):
        super().__init__(value)

    @staticmethod
    def get_wasm_type():
        return "f32"

    @staticmethod
    def get_random_val():
        return F32(np.float32(random.uniform(-2 ** 63, 2 ** 63 - 1)))

    @staticmethod
    def get_wasmtime_type():
        return wasmtime.ValType.f32()

    @staticmethod
    def get_default_value():
        return F32(np.float32(0.0))

    def to_init_str(self):
        return f"f32.const {self.value}"

    def __str__(self):
        return "F32: " + str(self.value)

class F64(Num):

    def __init__(self, value=np.float64(0.0)):
        super().__init__(value)

    @staticmethod
    def get_wasm_type():
        return "f64"

    @staticmethod
    def get_random_val():
        return F64(np.float64(random.uniform(-2 ** 63, 2 ** 63 - 1)))

    @staticmethod
    def get_wasmtime_type():
        return wasmtime.ValType.f64()

    @staticmethod
    def get_default_value():
        return F64(np.float64(0.0))

    def to_init_str(self):
        return f"f64.const {self.value}"

    def __str__(self):
        return "F64: " + str(self.value)
#Vector types
class Vec(Val):

    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        return "Vec: " + str(self.value)

class V128(Vec):

    def __init__(self, value=0):
        super().__init__(value)

    def __str__(self):
        return "V128: " + str(self.value)
#Reference types
class Ref(Val):

    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        return "Ref: " + str(self.value)

class RefFunc(Ref):

    def __init__(self, value: Union["Function", None]):
        super().__init__(value)

    def __str__(self):
        return "RefFunc: " + str(self.value)

    @staticmethod
    def get_wasm_type():
        return "funcref"

    @staticmethod
    def get_random_val():
        return RefFunc(None)

    @staticmethod
    def get_wasmtime_type():
        return wasmtime.ValType.funcref()

    @staticmethod
    def get_default_value():
        return RefFunc(None)

    def to_init_str(self):
        return f"ref.null func"

class RefExtern(Ref):

    def __init__(self, value=None):
        super().__init__(value)

    def __str__(self):
        return "RefExtern: " + str(self.value)

    @staticmethod
    def get_wasm_type():
        return "externref"

    @staticmethod
    def get_random_val():
        return RefExtern(None)

    @staticmethod
    def get_wasmtime_type():
        return wasmtime.ValType.externref()

    @staticmethod
    def get_default_value():
        return RefExtern(None)

    def to_init_str(self):
        return f"ref.null extern"


def get_random_val():
    """Returns a random val with default value."""
    return random.choice([I32, I64, F32, F64]).get_default_value()


def get_random_random_val():
    """Returns a random val with random value."""
    return random.choice([I32, I64, F32, F64]).get_random_val()
