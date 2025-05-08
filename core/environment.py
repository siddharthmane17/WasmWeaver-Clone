import threading
from typing import Any, SupportsFloat, List, Type

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box, Discrete

from core.constraints import AbstractConstraint, ByteCodeSizeConstraint, FuelConstraint, ConstraintsViolatedError
from core.loader import TileLoader
from core.state.functions import Function, Block
from drl.embedder.constraints import ConstraintsEmbedder
from core.value import Val
from core.state.state import GlobalState
from core.strategy import AbstractSelectionStrategy
from core.tile import AbstractTile
from core.util import generate_function
from drl.embedder.function import FunctionEmbedder
from drl.embedder.stack import StackEmbedder
from drl.embedder.tiles import TilesEmbedder

from torch.utils.tensorboard import SummaryWriter

EXCEPTION = -1
FINISHED_SUCCESS = 1

class EnvSelectionStrategy(AbstractSelectionStrategy):
    def __init__(self, env: "WasmWeaverEnv"):
        self.env = env

    name = "EnvSelectionStrategy"

    def get_weight(self, tile: Type["AbstractTile"], current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        if self.env.finish_thread:
            raise Exception("Thread was reset!")
        self.env.current_state = current_state
        self.env.current_function = current_function
        self.env.current_blocks = current_blocks
        self.env.current_tile = tile
        self.env.global_state_ready.release()
        self.env.action_ready.acquire()
        if tile == self.env.selected_tile:
            return 1.0
        return 0.0

class WasmWeaverEnv(gym.Env):
    def __init__(self, constraints: List[AbstractConstraint], input_types: List[Type[Val]] = None, output_types: List[Val] = None):
        super(WasmWeaverEnv, self).__init__()

        self.function_embedder = FunctionEmbedder()
        self.stack_embedder = StackEmbedder()
        self.constraints_embedder = ConstraintsEmbedder()
        self.tiles_embedder = TilesEmbedder()

        self.tile_loader = TileLoader("core/instructions/")
        self.tiles = list(self.tile_loader.tiles)

        self.input_types = input_types or []
        self.output_types = output_types or []

        self.action_space = Discrete(len(self.tiles))

        function_space = self.function_embedder.get_space()
        stack_space = self.stack_embedder.get_space()
        constraints_space = self.constraints_embedder.get_space()
        tile_space = self.tiles_embedder.get_space()

        flat_dim = (
            stack_space.shape[0] +
            function_space.shape[0] +
            constraints_space.shape[0] +
            tile_space.shape[0]
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

        self.constraints: List[AbstractConstraint] = constraints
        self.global_state_ready = threading.Semaphore(0)
        self.action_ready = threading.Semaphore(0)

        self.current_state: GlobalState | None = None
        self.current_tile: Type[AbstractTile] | None = None
        self.current_function: Function | None = None
        self.current_blocks: List[Block] | None = None

        self.selected_tile: Type[AbstractTile] | None = None
        self.selected_weight: float = 0
        self.thread: threading.Thread | None = None
        self.finish_state = None
        self.finish_thread = False

        self.current_step = 0  # Track step count

        # TensorBoard logging for stack/tile metrics
        self.writer = SummaryWriter(log_dir="./wasmweaver_tensorboard/env_metrics")
        self.episode_stack_sizes = []
        self.episode_tile_counts = {}
        self.episode_counter = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['thread']
        del state['global_state_ready']
        del state['action_ready']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.global_state_ready = threading.Semaphore(0)
        self.action_ready = threading.Semaphore(0)
        self.thread = None

    def generate(self):
        try:
            generate_function(
                self.tile_loader,
                "run",
                self.input_types,
                False,
                self.init_state,
                is_entry=True,
                selection_strategy=EnvSelectionStrategy(self),
                fixed_output_types=self.output_types
            )
            self.finish_state = "Success"
        except ConstraintsViolatedError as e:
            self.finish_state = e
        except Exception as e:
            self.finish_state = e
            raise e
        finally:
            self.global_state_ready.release()

    def _init_state(self):
        if self.thread is not None and self.thread.is_alive():
            self.finish_thread = True
            self.global_state_ready.release()
            self.thread.join(timeout=1)
            self.finish_thread = False

        self.init_state = GlobalState()
        for constraint in self.constraints:
            constraint.reset()
            self.init_state.constraints.add(constraint)

        self.init_state.stack.push_frame(params=None, stack=[], name="origin")
        self.selected_tile = None
        self.selected_weight = 0
        self.finish_state = None

        self.global_state_ready = threading.Semaphore(0)
        self.action_ready = threading.Semaphore(0)

        self.thread = threading.Thread(target=self.generate, daemon=True)
        self.thread.start()

    def _get_flat_obs(self):
        return np.concatenate([
            self.stack_embedder(self.current_state.stack),
            self.function_embedder(self.current_function),
            self.constraints_embedder(self.current_state.constraints.constraints),
            self.tiles_embedder(self.current_tile)
        ]).astype(np.float32)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.selected_tile = self.tiles[action]
        self.action_ready.release()
        self.global_state_ready.acquire()

        self.current_step += 1

        done = False
        truncated = False
        reward = 0.0

        if isinstance(self.finish_state, Exception):
            reward = -1.0
            done = True
        elif self.finish_state == "Success":
            reward = 1.0
            done = True
        else:
            # Reward shaping for learning signals
            if self.selected_tile.name != "NoOp":
                reward += 0.01
            if len(self.current_state.stack.get_current_frame().stack) > 0:
                reward += 0.02
            if self.selected_tile.name == "NoOp":
                reward -= 0.01

        # Track metrics
        stack_size = len(self.current_state.stack.get_current_frame().stack)

        self.episode_stack_sizes.append(stack_size)

        tile_name = self.selected_tile.name
        self.episode_tile_counts[tile_name] = self.episode_tile_counts.get(tile_name, 0) + 1

        if done:
            avg_stack = np.mean(self.episode_stack_sizes)
            self.writer.add_scalar("env/avg_stack_size", avg_stack, self.episode_counter)

            for tile, count in self.episode_tile_counts.items():
                self.writer.add_scalar(f"tiles/{tile}", count, self.episode_counter)

            self.episode_counter += 1
            self.episode_stack_sizes = []
            self.episode_tile_counts = {}

        obs = self._get_flat_obs()

        info = {}
        if done:
            info["episode"] = {
                "r": reward,
                "l": self.current_step
            }

        return obs, reward, done, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self._init_state()
        self.current_step = 0
        self.global_state_ready.acquire()
        return self._get_flat_obs(), {}
