import random
import struct

import numpy as np

from core.config.config import MEMORY_MAX_WRITE_INDEX


class Memory:

    def __init__(self, initial=1, maximum=1, index=0, default_page_size=65536):
        self.size = initial
        self.initial = initial
        self.maximum = maximum
        self.memory = bytearray(self.size * default_page_size)
        self.index = index
        #self.randomize() # Commented out to make the memory deterministic
        self.initial_values = bytearray(self.memory)

    def i32_store(self, offset, value):
        if offset < 0 or offset+4 > MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        struct.pack_into('<I', self.memory, offset, value & 0xFFFFFFFF)  # Store as unsigned 32-bit

    def i32_store8(self, offset, value):
        if offset < 0 or offset >= MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        #print("Storing 8 bit value", value)
        struct.pack_into('<B', self.memory, offset, value & 0xFF)

    def i32_store16(self, offset, value):
        if offset < 0 or offset + 2 > MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        struct.pack_into('<H', self.memory, offset, value & 0xFFFF)

    def i32_load(self, offset):
        if offset < 0 or offset + 4 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        return np.uint32(struct.unpack_from('<I', self.memory, offset)[0])  # Load as unsigned 32-bit

    def i32_load8_s(self, offset):
        if offset < 0 or offset >= MEMORY_MAX_WRITE_INDEX:
            raise IndexError(f"Offset out of bounds {offset}")
        value, = struct.unpack_from('<b', self.memory, offset)  # Load one byte and sign-extend
        return np.int32(value)

    def i32_load8_u(self, offset):
        if offset < 0 or offset >= MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<B', self.memory, offset)  # Load one byte and zero-extend
        return np.uint32(value)

    def i32_load16_s(self, offset):
        if offset < 0 or offset + 2 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<h', self.memory, offset)  # Load two bytes and sign-extend
        return np.int32(value)

    def i32_load16_u(self, offset):
        if offset < 0 or offset + 2 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<H', self.memory, offset)  # Load two bytes and zero-extend
        return np.uint32(value)

    def i64_store(self, offset, value):
        if offset < 0 or offset+8 > MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        struct.pack_into('<Q', self.memory, offset, value & 0xFFFFFFFFFFFFFFFF)

    def i64_store8(self, offset, value):
        if offset < 0 or offset >= MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        struct.pack_into('<B', self.memory, offset, value & 0xFF)

    def i64_store16(self, offset, value):
        if offset < 0 or offset + 2 > MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        struct.pack_into('<H', self.memory, offset, value & 0xFFFF)

    def i64_store32(self, offset, value):
        if offset < 0 or offset + 4 > MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        struct.pack_into('<I', self.memory, offset, value & 0xFFFFFFFF)

    def i64_load(self, offset):
        if offset < 0 or offset + 8 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        return np.uint64(struct.unpack_from('<Q', self.memory, offset)[0])

    def i64_load8_s(self, offset):
        if offset < 0 or offset >= MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<b', self.memory, offset)

        return np.int64(value)

    def i64_load8_u(self, offset):
        if offset < 0 or offset >= MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<B', self.memory, offset)

        return np.uint64(value)

    def i64_load16_s(self, offset):
        if offset < 0 or offset + 2 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<h', self.memory, offset)

        return np.int64(value)

    def i64_load16_u(self, offset):
        if offset < 0 or offset + 2 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<H', self.memory, offset)

        return np.uint64(value)

    def i64_load32_s(self, offset):
        if offset < 0 or offset + 4 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<i', self.memory, offset)

        return np.int64(value)

    def i64_load32_u(self, offset):
        if offset < 0 or offset + 4 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        value, = struct.unpack_from('<I', self.memory, offset)

        return np.uint64(value)

    def f32_store(self, offset, value):
        if offset < 0 or offset+4 > MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        struct.pack_into('<f', self.memory, offset, value)

    def f32_load(self, offset):
        if offset < 0 or offset+4 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        return np.float32(struct.unpack_from('<f', self.memory, offset)[0])

    def f64_store(self, offset, value):
        if offset < 0 or offset+8 > MEMORY_MAX_WRITE_INDEX:
            raise ValueError("Memory index out of bounds")
        struct.pack_into('<d', self.memory, offset, value)

    def f64_load(self, offset):
        if offset < 0 or offset+8 > MEMORY_MAX_WRITE_INDEX:
            raise IndexError("Offset out of bounds")
        return np.float64(struct.unpack_from('<d', self.memory, offset)[0])


    def __getitem__(self, index):
        return self.memory[index]

    def __setitem__(self, index, value):
        self.memory[index] = value

    def __str__(self):
        """Return a string representation of the memory. First MEMORY_MAX_WRITE_INDEX bytes are shown as hex values. 64 bytes per line."""
        BYTES_PER_ROW = 32
        return "\n".join([f"{i:04x}: {' '.join([f'{self.memory[i + j]:02x}' for j in range(BYTES_PER_ROW)])}" for i in
                          range(0, MEMORY_MAX_WRITE_INDEX, BYTES_PER_ROW)])

    def randomize(self):
        self.memory = bytearray(random.choices(range(256), k=len(self.memory)))

    def reinit_memory(self):
        """Re-initializes the memory to its initial values."""
        self.memory = bytearray(self.initial_values)
