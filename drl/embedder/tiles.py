import numpy as np
from gymnasium.spaces import Box
from typing import List, Type
from core.tile import AbstractTile

MAX_TILE_ID = 512
MAX_ARG_VALUE = 128

class TilesEmbedder:

    def __init__(self):
        self.max_id = MAX_TILE_ID
        self.max_arg = MAX_ARG_VALUE

    def get_space(self):
        return Box(low=0.0, high=1.0, shape=(self.max_id + 1,), dtype=np.float32)

    def __call__(self, tile: AbstractTile):
        tile_id = self.get_id(tile)
        tile_id_onehot = np.zeros(self.max_id, dtype=np.float32)
        tile_id_onehot[tile_id] = 1.0

        tile_arg = self.get_args(tile) / self.max_arg
        return np.concatenate([tile_id_onehot, [tile_arg]]).astype(np.float32)

    def get_id(self, tile: AbstractTile | Type[AbstractTile]):
        name = tile.name
        match name:
            case "F32Const": return 80
            case "F64Const": return 120
            case "I32Const": return 160
            case "Get global": return 290
            case "Finish": return 5
            case "NoOp": return 10
            case "Drop": return 11
            case "Select": return 12
            case "I64Const": return 220
            case "F32Add": return 81
            case "F64Mul": return 123
            case "I32Sub": return 162
            case "Get local": return 280
            case "Set global": return 291
            case "Memory size": return 300
            case "Get table": return 310
            case _: 
                print(f"[WARN] Unknown tile name: {name}. Returning fallback ID 0.")
                return 0

    def get_args(self, tile: AbstractTile | Type[AbstractTile]):
        match tile.name:
            case "Create and call function" | "Call function" | "Indirect call function" | "Push function reference to stack":
                return getattr(tile, "index", 0)
            case "Get global" | "Set global":
                return getattr(tile, "index", 0)
            case "Get local" | "Set local" | "Tee local":
                return getattr(tile, "index", 0)
            case "Get table" | "Set table":
                return getattr(tile, "index", 0)
            case _:
                return 0