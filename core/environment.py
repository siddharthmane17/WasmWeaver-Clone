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

from core.constraints import AbstractConstraint, ByteCodeSizeConstraint, FuelConstraint
from core.loader import TileLoader
from core.state.functions import Function
from core.state.state import GlobalState
from core.strategy import AbstractSelectionStrategy
from core.tile import AbstractTile
from core.util import generate_function

EXCEPTION = -1
FINISHED_SUCCESS = 1


class EnvSelectionStrategy(AbstractSelectionStrategy):
    """
    Random selection strategy.
    """

    def __init__(self, env: "WasmWeaverEnv"):
        self.env = env

    name = "EnvSelectionStrategy"  # This is the name of the strategy

    def get_weight(self, tile: Type["AbstractTile"], current_state: GlobalState, current_function: Function):
        """
        Returns a random weight for the tile.
        """
        if self.env.finish_thread:
            raise Exception("Thread was reset!")
        self.env.current_state = current_state
        self.env.current_function = current_function
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

OBSERVATION_SPACE = Dict({"global_state": ObjectSpace(),
                                       "current_function": ObjectSpace(),
                                       "tile": ObjectSpace()})

ACTION_SPACE =  gym.spaces.Box(low=0, high=1, shape=(1,))

class WasmWeaverEnv(gym.Env):

    def __init__(self, config: EnvContext):
        super(WasmWeaverEnv, self).__init__()
        self.action_space =  ACTION_SPACE# The weight
        self.observation_space = OBSERVATION_SPACE
        self.constraints: List[AbstractConstraint] = config.get("constraints", [])
        self.global_state_ready = threading.Semaphore(0)
        self.action_ready = threading.Semaphore(0)
        self.current_state: GlobalState | None = None
        self.current_tile: Type[AbstractTile] | None = None
        self.current_function: Function | None = None
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
            print("Starting!")
            generate_function(self.tile_loader, "run", [], False, self.init_state,
                              is_entry=True, selection_strategy=EnvSelectionStrategy(self))
            self.finish_state = "Success"
            self.global_state_ready.release()
            print("Finished!")
        except Exception as e:
            print(e)
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
        return ({"global_state": np.array([self.current_state],dtype=object),
                 "current_function": np.array([self.current_function],dtype=object),
                 "tile": np.array([self.current_tile],dtype=object)},
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
        self.global_state_ready.acquire()

        return ({"global_state": np.array([self.current_state],dtype=object),
                 "current_function": np.array([self.current_function],dtype=object),
                 "tile": np.array([self.current_tile],dtype=object)},
                {})


def main():
    gym.register(
        id="gymnasium_env/WasmWeaverEnv-v0",
        entry_point=WasmWeaverEnv,
    )

    env = gym.make("gymnasium_env/WasmWeaverEnv-v0",
                   constraints=[ByteCodeSizeConstraint(0, 100), FuelConstraint(10, 5000)])
    print("Environment created")
    for epoch in range(100):
        done = False
        env.reset()
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done:
                print(f"Epoch: {epoch}, Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}")
                print("Done!")
                break


if __name__ == '__main__':
    main()
