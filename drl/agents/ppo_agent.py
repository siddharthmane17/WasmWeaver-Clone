from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from core.environment import WasmWeaverEnv
from core.constraints import FuelConstraint, ByteCodeSizeConstraint

# 1. Define constraints
constraints = [
    FuelConstraint(min_target=50, max_target=100, initial=100),
    ByteCodeSizeConstraint(min_target=100, max_target=200, initial=100)
]

# 2. Wrap your environment with DummyVecEnv
env = DummyVecEnv([lambda: WasmWeaverEnv(constraints=constraints)])

# 3. Create PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./wasmweaver_tensorboard/")


# 4. Train the agent
model.learn(total_timesteps=2000000)

# 5. Save the trained model
model.save("ppo_wasmweaver_model")
print("PPO agent trained and saved!")
