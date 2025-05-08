import numpy as np
from gymnasium.spaces import Box
import math
import sys
from core.config.config import MAX_STACK_SIZE
from core.state.stack import Stack
from core.value import I32, I64, F32, F64, RefFunc
from drl.embedder.values import embedd_value_type, MAX_VALUE_TYPE_INDEX

_LOG_MAX64 = math.log1p(sys.float_info.max)

def symlog_to_unit(x):
    x64 = np.asarray(x, dtype=np.float64)
    nan_mask = np.isnan(x64)
    inf_mask = np.isinf(x64)
    abs_clamped = np.minimum(np.abs(x64), sys.float_info.max)
    y = np.sign(x64) * np.log1p(abs_clamped) / _LOG_MAX64
    y = np.where(inf_mask, np.sign(x64), y)
    y = np.where(nan_mask, 0.0, y)
    return y.astype(np.float32, copy=False)

class StackEmbedder:
    def __init__(self):
        self.stack_size = MAX_STACK_SIZE
        self.stack_embed_dim = 4  # Now 4 values per entry
        self.flat_dim = self.stack_embed_dim * self.stack_size

    def get_space(self):
        return Box(low=-1, high=1, shape=(self.flat_dim,), dtype=np.float32)

    def __call__(self, stack: Stack):
        current_stack_frame = stack.get_current_frame()
        stack_values = current_stack_frame.stack

        id_tensor = np.zeros(self.stack_size, dtype=np.float32)
        values_tensor = np.zeros(self.stack_size, dtype=np.float32)
        mask_tensor = np.zeros(self.stack_size, dtype=np.float32)
        pad_tensor = np.zeros(self.stack_size, dtype=np.float32)  # Placeholder padding

        for i, value in enumerate(stack_values):
            if i >= self.stack_size:
                break
            id_tensor[i] = embedd_value_type(value) / MAX_VALUE_TYPE_INDEX
            mask_tensor[i] = 1.0

            if isinstance(value, (I32, I64, F32, F64)):
                values_tensor[i] = symlog_to_unit(value.value)
            elif isinstance(value, RefFunc):
                values_tensor[i] = -1.0 if value.value is None else float(value.value.index)
            else:
                raise ValueError(f"Unknown value type: {type(value)}")

            pad_tensor[i] = 0.0  # could later be type_flag, etc.

        # Stack shape: [id, value, mask, pad] × 32 = 128 total
        return np.concatenate([id_tensor, values_tensor, mask_tensor, pad_tensor]).astype(np.float32)
