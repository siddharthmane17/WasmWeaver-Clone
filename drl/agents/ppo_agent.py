from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from drl.agents.feature_extractor import WasmFeatureExtractor
from core.environment import WasmWeaverEnv
from core.constraints import FuelConstraint, ByteCodeSizeConstraint

# 1. Define constraints
constraints = [
    FuelConstraint(min_target=50, max_target=100, initial=100),
    ByteCodeSizeConstraint(min_target=100, max_target=200, initial=100)
]

# 2. Wrapping the environment in Monitor for TensorBoard logging
env = DummyVecEnv([lambda: Monitor(WasmWeaverEnv(constraints=constraints))])

# 3. Define policy_kwargs for custom feature extractor
policy_kwargs = dict(
    features_extractor_class=WasmFeatureExtractor,
    features_extractor_kwargs=dict(
        max_tile_id=512,
        stack_size=32,             # Match MAX_STACK_SIZE
        stack_embed_dim=4          # Updated for Transformer + LSTM
    ),
)


model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./wasmweaver_tensorboard/",
    policy_kwargs=policy_kwargs,
)

# 5. Train the agent
model.learn(total_timesteps=200_00)  

# 6. Saving trained model
model.save("ppo_wasmweaver_model")
print("PPO agent trained and saved!")
