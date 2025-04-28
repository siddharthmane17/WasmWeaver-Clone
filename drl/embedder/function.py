import numpy as np
from gymnasium.spaces import Dict, Discrete, Box, MultiDiscrete

from core.config.config import MAX_FUNCTION_INPUTS, MAX_FUNCTION_OUTPUTS, MAX_FUNCTIONS_PER_MODULE
from core.state.functions import Function
from core.value import I32, I64, F32, F64
from drl.embedder.values import embedd_value_type, MAX_VALUE_TYPE_INDEX


class FunctionEmbedder:

    def __init__(self):
        ...

    def get_space(self):
        return Dict({
            "index": Discrete(MAX_FUNCTIONS_PER_MODULE),
            "inputs": MultiDiscrete([MAX_VALUE_TYPE_INDEX+1]*MAX_FUNCTION_INPUTS),
            "outputs": MultiDiscrete([MAX_VALUE_TYPE_INDEX+1]*MAX_FUNCTION_OUTPUTS),
            "inputs_mask": MultiDiscrete([2]*MAX_FUNCTION_INPUTS),
            "outputs_mask": MultiDiscrete([2]*MAX_FUNCTION_OUTPUTS)
        })

    def __call__(self, function: Function):
        index = function.index
        inputs_tensor = np.zeros(MAX_FUNCTION_INPUTS)
        outputs_tensor = np.zeros(MAX_FUNCTION_OUTPUTS)
        inputs_mask = np.zeros(MAX_FUNCTION_INPUTS)
        outputs_mask = np.zeros(MAX_FUNCTION_OUTPUTS)
        for i, input_type in enumerate(function.inputs):
            inputs_tensor[i] = embedd_value_type(input_type.get_default_value())
            inputs_mask[i] = 1

        for i, output_type in enumerate(function.outputs):
            outputs_tensor[i] = embedd_value_type(output_type.get_default_value())
            outputs_mask[i] = 1
        return {
            "index": index,
            "inputs": inputs_tensor,
            "outputs": outputs_tensor,
            "inputs_mask": inputs_mask,
            "outputs_mask": outputs_mask
        }

if __name__ == "__main__":
    function = Function(index=1, inputs=[I32, I64], outputs=[F32, F64],name="test_function")
    embedder = FunctionEmbedder()
    embedding = embedder(function)
    print(embedding)
    print(embedder.get_space().contains(embedding))
    print(embedder.get_space())
