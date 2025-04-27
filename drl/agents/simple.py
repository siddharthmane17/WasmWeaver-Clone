import ray
import torch
import torch.nn as nn
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_action_dist import TorchBeta

from core.constraints import ByteCodeSizeConstraint, FuelConstraint
from core.environment import WasmWeaverEnv, ACTION_SPACE, OBSERVATION_SPACE
from drl.embedder.constraints import ConstraintsEmbedder


class CustomRLModule(TorchRLModule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Define your embedding layers and policy network
        print("Inference only:",args,kwargs)
        self.constraint_embedding = ConstraintsEmbedder()
        self.trunk = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
        )

        # ---------- actor head: α, β > 0 ----------
        self.actor = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softplus()  # strictly positive outputs
        )

        # ---------- critic head ----------
        self.critic = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward_exploration(self, batch, **kwargs):
        return self._forward(batch)

    def forward_inference(self, batch, **kwargs):
        return self._forward(batch)

    def forward_train(self, batch, **kwargs):
        return self._forward(batch)

    def _forward(self, batch):
        # Extract and process the custom object

        current_functions = batch["obs"]["current_function"]
        global_states = batch["obs"]["global_state"]
        tiles = batch["obs"]["tile"]

        constraint_embeddings = []
        for i in range(len(global_states)):
            constraints = global_states[i][0].constraints.constraints
            embedding = self.constraint_embedding(constraints)
            constraint_embeddings.append(embedding)

        emb = torch.stack(constraint_embeddings)
        hidden = self.trunk(emb)  # (B, 128)

        # Actor: α and β (add small ε to avoid zeros)
        alpha_beta = self.actor(hidden) + 1e-4  # (B, 2)

        # Critic: scalar value
        value = self.critic(emb).squeeze(-1)
        return {
            "action_dist_inputs": alpha_beta,
            "action_dist_class": "TorchBeta",
            Columns.VF_PREDS: value,
        }




def main():
    ray.init(local_mode=True,include_dashboard=False,num_cpus=1)
    constraints = [ByteCodeSizeConstraint(0, 50), FuelConstraint(10, 5000)]

    class NoMaskPPOLearner(PPOTorchLearner):
        def compute_loss_for_module(self, *a, **kw):
            kw["batch"].pop(Columns.LOSS_MASK, None)  # drop if present
            return super().compute_loss_for_module(*a, **kw)
    cfg = (
        PPOConfig()
        .environment(env=WasmWeaverEnv,
                     env_config={
                         "constraints":constraints,
                     },
                     action_space=ACTION_SPACE,
                    observation_space=OBSERVATION_SPACE,
                     disable_env_checking=True,
                     normalize_actions=False,
                     clip_actions=False)
        .rl_module(
            rl_module_spec=RLModuleSpec(
            module_class=CustomRLModule,
            observation_space=OBSERVATION_SPACE,
            action_space=ACTION_SPACE,
        )
        )
        .framework("torch")
        .training(lr=3e-4, gamma=0.99,learner_class=NoMaskPPOLearner)
    )
    cfg.env_runners(num_env_runners=1)
    algo = cfg.build_algo(use_copy=False)
    for _ in range(200):
        print(algo.train()["episode_reward_mean"])


if __name__ == '__main__':
    main()