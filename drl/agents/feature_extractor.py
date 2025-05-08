import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class WasmFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, max_tile_id=512, stack_size=32, stack_embed_dim=4):
        super().__init__(observation_space, features_dim=128)

        self.total_input_dim = observation_space.shape[0]

        self.max_tile_id = max_tile_id
        self.stack_size = stack_size
        self.stack_embed_dim = stack_embed_dim
        self.stack_dim = self.stack_size * self.stack_embed_dim

        # Transformer Encoder (local dependencies in stack)
        self.stack_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.stack_embed_dim,
                nhead=2,
                batch_first=True
            ),
            num_layers=1
        )

        # LSTM for temporal features (longer dependencies)
        self.stack_lstm = nn.LSTM(
            input_size=self.stack_embed_dim,
            hidden_size=32,
            batch_first=True
        )

        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(32 + (self.total_input_dim - self.stack_dim), 256),
            nn.ReLU(),
            nn.Linear(256, self.features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split stack and rest
        stack = observations[:, :self.stack_dim]  # [B, 128]
        rest = observations[:, self.stack_dim:]   # [B, 743]

        # Reshape to [B, 32, 4]
        stack_seq = stack.view(-1, self.stack_size, self.stack_embed_dim)

        # Pass through Transformer
        stack_encoded = self.stack_transformer(stack_seq)  # [B, 32, 4]

        # Pass through LSTM
        _, (h_n, _) = self.stack_lstm(stack_encoded)       # h_n: [1, B, 32]
        stack_feat = h_n.squeeze(0)                        # [B, 32]

        # Concatenate and project
        full_input = torch.cat([stack_feat, rest], dim=1)  # [B, 775]
        return self.projection(full_input)                 # [B, 128]
