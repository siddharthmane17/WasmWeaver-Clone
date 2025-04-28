import numpy as np
from gymnasium.spaces import Dict, MultiDiscrete, Box
import math
import sys
from core.config.config import MAX_STACK_SIZE
from core.state.stack import Stack, StackFrame
from core.value import I32, I64, F32, F64, RefFunc
from drl.embedder.values import embedd_value_type, MAX_VALUE_TYPE_INDEX

_LOG_MAX64 = math.log1p(sys.float_info.max)  # ≈ 709.7827  (float64, well inside range)


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
        ...

    def get_space(self):
        return Dict({
            "ids": MultiDiscrete([MAX_VALUE_TYPE_INDEX+1]*MAX_STACK_SIZE, dtype=np.int32),
            "args": Box(low=-1, high=100, shape=(MAX_STACK_SIZE,), dtype=np.float32),
            "mask": MultiDiscrete([2]*MAX_STACK_SIZE, dtype=np.int32)
        })

    def __call__(self, stack: Stack):
        current_stack_frame = stack.get_current_frame()
        stack_values = current_stack_frame.stack
        id_tensor = np.zeros(MAX_STACK_SIZE, dtype=np.int32)
        values_tensor = np.zeros(MAX_STACK_SIZE, dtype=np.float32)
        mask = np.zeros(MAX_STACK_SIZE, dtype=np.int32)
        for i, value in enumerate(stack_values):
            id_tensor[i] = embedd_value_type(value)
            if isinstance(value, I32):
                values_tensor[i] = symlog_to_unit(value.value)
                mask[i] = 1
            elif isinstance(value, I64):
                values_tensor[i] = symlog_to_unit(value.value)
                mask[i] = 1
            elif isinstance(value, F32):
                values_tensor[i] = symlog_to_unit(value.value)
                mask[i] = 1
            elif isinstance(value, F64):
                values_tensor[i] = symlog_to_unit(value.value)
                mask[i] = 1
            elif isinstance(value, RefFunc):
                if value.value is None:
                    values_tensor[i] = -1
                else:
                    values_tensor[i] = value.value.index #Its a function
                mask[i] = 1
            else:
                raise ValueError(f"Unknown value type: {type(value)}")

        return {"ids": id_tensor, "args": values_tensor, "mask": mask}

if __name__ == "__main__":
    stack = Stack()
    stack.stack_frames.append(StackFrame())
    stack.stack_frames[0].stack_push(I32(1))
    stack.stack_frames[0].stack_push(I64(2))
    stack.stack_frames[0].stack_push(F32(3.0))
    stack.stack_frames[0].stack_push(F64(4.0))

    stack_embedder = StackEmbedder()
    embedding = stack_embedder(stack)
    print(stack_embedder.get_space().contains(embedding))
    print(stack_embedder.get_space().sample())
    print(embedding)

