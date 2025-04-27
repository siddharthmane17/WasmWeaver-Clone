# any_space_rollout_buffer.py
"""
Rollout buffer for Stable-Baselines 3 that accepts *arbitrary* Python
objects (dtype=object) as observations and actions.

* Numeric data (rewards, values, advantages, returns) stay in the usual
  NumPy / Torch arrays → GAE and PPO losses remain unchanged.
* Observations and actions are stored in ordinary Python lists of shape
  [buffer_size][n_envs].  Your **policy** is responsible for turning
  those objects into tensors inside `forward()`.
* Slower than the default buffer because object data cannot live on the
  GPU.  Use only when needed.

Tested with SB-3 >= 2.3.
"""

from __future__ import annotations

from typing import Any, Generator, List, Optional

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples


class AnySpaceRolloutBuffer(RolloutBuffer):
    """A RolloutBuffer that stores raw Python objects for obs/actions."""

    # --------------------------------------------------------------------- #
    # construction                                                          #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: str | th.device = "cpu",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

        # Python-object storage
        self.raw_obs: List[List[Any]] = [[None] * n_envs for _ in range(buffer_size)]
        self.raw_actions: List[List[Any]] = [[None] * n_envs for _ in range(buffer_size)]

    # --------------------------------------------------------------------- #
    # store one transition                                                  #
    # --------------------------------------------------------------------- #
    def add(
        self,
        obs: List[Any],        # list length n_envs, arbitrary objects allowed
        action: List[Any],     # same length, arbitrary objects
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if self.full:
            raise ValueError("RolloutBuffer overflow — call .reset()")

        self.raw_obs[self.pos] = list(obs)
        self.raw_actions[self.pos] = list(action)

        # numeric arrays (copied from original RolloutBuffer.add)
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.detach().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy().flatten()

        # advance pointer
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    # --------------------------------------------------------------------- #
    # sample mini-batches                                                   #
    # --------------------------------------------------------------------- #
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        if not self.full:
            raise ValueError(
                "Rollout buffer must be full before sampling — collect more steps first."
            )

        total_steps = self.buffer_size * self.n_envs
        inds = np.random.permutation(total_steps)
        batch_size = batch_size or total_steps

        for start in range(0, total_steps, batch_size):
            batch_inds = inds[start : start + batch_size]

            # flat index → (time, env) pairs
            t_idx = batch_inds // self.n_envs
            e_idx = batch_inds % self.n_envs

            obs_batch = [self.raw_obs[t][e] for t, e in zip(t_idx, e_idx)]
            act_batch = [self.raw_actions[t][e] for t, e in zip(t_idx, e_idx)]

            values = th.as_tensor(self.values[t_idx, e_idx], device=self.device)
            logp   = th.as_tensor(self.log_probs[t_idx, e_idx], device=self.device)
            adv    = th.as_tensor(self.advantages[t_idx, e_idx], device=self.device)
            rets   = th.as_tensor(self.returns[t_idx, e_idx], device=self.device)

            yield RolloutBufferSamples(
                observations=obs_batch,       # type: ignore[arg-type]
                actions=act_batch,            # type: ignore[arg-type]
                values=values.squeeze(-1),
                log_probs=logp.squeeze(-1),
                advantages=adv.squeeze(-1),
                returns=rets.squeeze(-1),
            )

    # --------------------------------------------------------------------- #
    def reset(self) -> None:
        super().reset()
        for t in range(self.buffer_size):
            for e in range(self.n_envs):
                self.raw_obs[t][e] = None
                self.raw_actions[t][e] = None
