import torch
import numpy as np
from gymnasium.spaces import Box
from drl.agents.feature_extractor import WasmFeatureExtractor

def test_feature_extractor_forward_pass():
    stack_size = 32
    stack_embed_dim = 3
    tile_dim = 513  # One-hot tile + normalized arg
    function_dim = 129
    constraint_dim = 100

    total_obs_dim = stack_size * stack_embed_dim + function_dim + constraint_dim + tile_dim

    obs_space = Box(low=-1, high=1, shape=(total_obs_dim,), dtype=np.float32)
    feature_extractor = WasmFeatureExtractor(obs_space, stack_size=stack_size, stack_embed_dim=stack_embed_dim)

    dummy_obs = torch.rand((2, total_obs_dim), dtype=torch.float32)  # batch of 2
    output = feature_extractor(dummy_obs)

    assert output.shape == (2, feature_extractor.features_dim), \
        f"Expected shape (2, {feature_extractor.features_dim}), got {output.shape}"
    
    print("Feature extractor test passed!")
    print("Input shape :", dummy_obs.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_feature_extractor_forward_pass()
