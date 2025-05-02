import numpy as np
from gymnasium.spaces import Box
from core.config.config import MAX_FUNCTION_INPUTS, MAX_FUNCTION_OUTPUTS, MAX_FUNCTIONS_PER_MODULE
from core.state.functions import Function
from core.value import I32, I64, F32, F64
from drl.embedder.values import embedd_value_type, MAX_VALUE_TYPE_INDEX


class FunctionEmbedder:

    def __init__(self):
        self.input_size = MAX_FUNCTION_INPUTS
        self.output_size = MAX_FUNCTION_OUTPUTS

        # 1 for index, rest for inputs/outputs/masks
        self.flat_dim = 1 + self.input_size + self.output_size + self.input_size + self.output_size

    def get_space(self):
        # A flat Box containing all components
        return Box(low=0, high=1, shape=(self.flat_dim,), dtype=np.float32)

    def __call__(self, function: Function):
        # 1. Normalize index
        index = np.array([function.index / MAX_FUNCTIONS_PER_MODULE], dtype=np.float32)

        # 2. Encode inputs and mask
        inputs_tensor = np.zeros(self.input_size, dtype=np.float32)
        inputs_mask = np.zeros(self.input_size, dtype=np.float32)
        for i, input_type in enumerate(function.inputs):
            inputs_tensor[i] = embedd_value_type(input_type.get_default_value()) / MAX_VALUE_TYPE_INDEX
            inputs_mask[i] = 1.0

        # 3. Encode outputs and mask
        outputs_tensor = np.zeros(self.output_size, dtype=np.float32)
        outputs_mask = np.zeros(self.output_size, dtype=np.float32)
        for i, output_type in enumerate(function.outputs):
            outputs_tensor[i] = embedd_value_type(output_type.get_default_value()) / MAX_VALUE_TYPE_INDEX
            outputs_mask[i] = 1.0

        # 4. Concatenate everything into one flat array
        return np.concatenate([
            index,
            inputs_tensor,
            outputs_tensor,
            inputs_mask,
            outputs_mask
        ]).astype(np.float32)


if __name__ == "__main__":
    function = Function(index=1, inputs=[I32, I64], outputs=[F32, F64], name="test_function")
    embedder = FunctionEmbedder()
    embedding = embedder(function)
    print(embedding)
    print(embedder.get_space().contains(embedding))
    print(embedder.get_space())
