import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class WasmFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, max_tile_id=512, stack_size=32, stack_embed_dim=3):
        super().__init__(observation_space, features_dim=128)

        self.total_input_dim = observation_space.shape[0]

        self.max_tile_id = max_tile_id
        self.stack_size = stack_size
        self.stack_embed_dim = stack_embed_dim
        self.stack_dim = self.stack_size * self.stack_embed_dim

        # Transformer encoder for the stack
        self.stack_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.stack_embed_dim, nhead=1, batch_first=True
            ),
            num_layers=1
        )

        self.projector = nn.Sequential(
            nn.Linear((self.total_input_dim - self.stack_dim) + self.stack_size * self.stack_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # [batch, stack_dim] → [batch, stack_size, 3]
        stack = observations[:, :self.stack_dim]
        stack = stack.view(-1, self.stack_size, self.stack_embed_dim)

        stack_encoded = self.stack_transformer(stack)
        stack_flat = stack_encoded.reshape(stack_encoded.size(0), -1)

        # Rest of the observation (function + constraints + tile)
        rest = observations[:, self.stack_dim:]

        final_obs = torch.cat([stack_flat, rest], dim=1)
        return self.projector(final_obs)
