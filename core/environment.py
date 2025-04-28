import random
import threading
from typing import Any, SupportsFloat, List, Type

import gymnasium as gym
import numpy as np
from gymnasium import Space
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Dict, Box
from gymnasium.spaces.space import T_cov
from ray.rllib.env import EnvContext

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

EXCEPTION = -1
FINISHED_SUCCESS = 1


class EnvSelectionStrategy(AbstractSelectionStrategy):
    """
    Random selection strategy.
    """

    def __init__(self, env: "WasmWeaverEnv"):
        self.env = env

    name = "EnvSelectionStrategy"  # This is the name of the strategy

    def get_weight(self, tile: Type["AbstractTile"], current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
        """
        Returns a random weight for the tile.
        """
        if self.env.finish_thread:
            raise Exception("Thread was reset!")
        self.env.current_state = current_state
        self.env.current_function = current_function
        self.env.current_blocks = current_blocks
        self.env.current_tile = tile
        self.env.global_state_ready.release()
        self.env.action_ready.acquire()
        return self.env.selected_weight


class ObjectSpace(Space):
    def __init__(self, cls: type | tuple[type, ...] | None = None):
        #  shape=()  → scalar (one object per env)
        #  dtype=object  → NumPy will store just the reference
        super().__init__(shape=(1,), dtype=np.object_)
        self._cls = cls

    def contains(self, x) -> bool:              # called by RLlib checks
        return self._cls is None or isinstance(x, self._cls)

    # Not used by RLlib but keeps the API complete
    def sample(self, mask: Any | None = None) -> T_cov:
        raise NotImplementedError("Cannot sample an arbitrary object.")

class WasmWeaverEnv(gym.Env):

    def __init__(self, constraints: List[AbstractConstraint], input_types: List[Type[Val]] = None, output_types: List[Val] = None):
        super(WasmWeaverEnv, self).__init__()

        self.function_embedder = FunctionEmbedder()
        self.stack_embedder = StackEmbedder()
        self.constraints_embedder = ConstraintsEmbedder()
        self.tiles_embedder = TilesEmbedder()

        if input_types is None:
            self.input_types = []
        else:
            self.input_types = input_types

        if output_types is None:
            self.output_types = []
        else:
            self.output_types = output_types

        self.action_space = gym.spaces.Box(low=0, high=1, shape=())
        self.observation_space = Dict({
           "current_function":self.function_embedder.get_space(),
           "current_stack":self.stack_embedder.get_space(),
           "constraints": self.constraints_embedder.get_space(),
           "tile":self.tiles_embedder.get_space()
        })
        self.constraints: List[AbstractConstraint] = constraints
        self.global_state_ready = threading.Semaphore(0)
        self.action_ready = threading.Semaphore(0)
        self.current_state: GlobalState | None = None
        self.current_tile: Type[AbstractTile] | None = None
        self.current_function: Function | None = None
        self.current_blocks: List[Block] | None = None
        self.selected_weight: float = 0
        self.thread: threading.Thread | None = None
        self.tile_loader = TileLoader("core/instructions/")
        self.finish_state = None
        self.finish_thread = False





    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the thread from the state
        del state['thread']
        del state['global_state_ready']
        del state['action_ready']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the semaphores
        self.global_state_ready = threading.Semaphore(0)
        self.action_ready = threading.Semaphore(0)
        # Recreate the thread
        self.thread = None

    def generate(self):
        try:
            generate_function(self.tile_loader,
                        "run",
                              self.input_types,
                              False,
                              self.init_state,
                              is_entry=True,
                              selection_strategy=EnvSelectionStrategy(self),
                              fixed_output_types=self.output_types
                              )
            self.finish_state = "Success"
            self.global_state_ready.release()
        except ConstraintsViolatedError as e:
            self.finish_state = e
            self.global_state_ready.release()
        except Exception as e:
            raise e
            self.finish_state = e
            self.global_state_ready.release()

    def _init_state(self):
        if self.thread is not None and self.thread.is_alive():
            self.finish_thread = True
            self.global_state_ready.release()
            self.thread.join(timeout=1)
            self.finish_thread = False
        self.init_state = GlobalState()
        for constraint in self.constraints:
            # Reset constraints
            constraint.reset()
            self.init_state.constraints.add(constraint)
        self.init_state.stack.push_frame(params=None, stack=[], name="origin")
        self.selected_weight = 0
        self.finish_state = None
        self.global_state_ready = threading.Semaphore(0)
        self.action_ready = threading.Semaphore(0)
        self.thread = threading.Thread(target=self.generate, daemon=True)
        self.thread.start()

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.selected_weight = float(action)
        self.action_ready.release()
        self.global_state_ready.acquire()
        done = False
        truncated = False
        reward = 0
        if isinstance(self.finish_state, Exception):
            reward = EXCEPTION
            done = True
        elif self.finish_state == "Success":
            reward = FINISHED_SUCCESS
            done = True

        state = {
           "current_function":self.function_embedder(self.current_function),
           "current_stack":self.stack_embedder(self.current_state.stack),
           "constraints": self.constraints_embedder(self.current_state.constraints.constraints),
           "tile":self.tiles_embedder(self.current_tile)
        }

        if not self.observation_space.contains(state):
            # Check all spaces
            if self.function_embedder.get_space().contains(self.function_embedder(self.current_function)):
                print("Function embedder ok!")
            else:
                print("Function embedder failed!")
                print(self.function_embedder(self.current_function))

            if self.stack_embedder.get_space().contains(self.stack_embedder(self.current_state.stack)):
                print("Stack embedder ok!")
            else:
                print("Stack embedder failed!")
                print(self.stack_embedder(self.current_state.stack))

            if self.constraints_embedder.get_space().contains(
                    self.constraints_embedder(self.current_state.constraints.constraints)):
                print("Constraints embedder ok!")
            else:
                print("Constraints embedder failed!")
                print(self.constraints_embedder(self.current_state.constraints.constraints))

            if self.tiles_embedder.get_space().contains(self.tiles_embedder(self.current_tile)):
                print("Tiles embedder ok!")
            else:
                print("Tiles embedder failed!")
                print(self.tiles_embedder(self.current_tile))

            raise Exception(f"Observation is not valid")

        return (state,
                reward,
                done,
                truncated,
                {})

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed)

        self._init_state()

        print(self.global_state_ready._value)

        self.global_state_ready.acquire()

        state = {
           "current_function":self.function_embedder(self.current_function),
           "current_stack":self.stack_embedder(self.current_state.stack),
           "constraints": self.constraints_embedder(self.current_state.constraints.constraints),
           "tile":self.tiles_embedder(self.current_tile)
        }

        if not self.observation_space.contains(state):
            #Check all spaces
            if self.function_embedder.get_space().contains(self.function_embedder(self.current_function)):
                print("Function embedder ok!")
            else:
                print("Function embedder failed!")
                print(self.function_embedder(self.current_function))

            if self.stack_embedder.get_space().contains(self.stack_embedder(self.current_state.stack)):
                print("Stack embedder ok!")
            else:
                print("Stack embedder failed!")
                print(self.stack_embedder(self.current_state.stack))

            if self.constraints_embedder.get_space().contains(self.constraints_embedder(self.current_state.constraints.constraints)):
                print("Constraints embedder ok!")
            else:
                print("Constraints embedder failed!")
                print(self.constraints_embedder(self.current_state.constraints.constraints))

            if self.tiles_embedder.get_space().contains(self.tiles_embedder(self.current_tile)):
                print("Tiles embedder ok!")
            else:
                print("Tiles embedder failed!")
                print(self.tiles_embedder(self.current_tile))

            raise Exception(f"Observation is not valid")

        return (state,
                {})


