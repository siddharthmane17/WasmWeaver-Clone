#A simple random agent to show how to interact with the environment
import gymnasium as gym
from core.environment import WasmWeaverEnv
from core.constraints import ByteCodeSizeConstraint, FuelConstraint

def main():
    gym.register(
        id="gymnasium_env/WasmWeaverEnv-v0",
        entry_point=WasmWeaverEnv,
    )

    env = gym.make("gymnasium_env/WasmWeaverEnv-v0",
                   constraints=[ByteCodeSizeConstraint(0, 100), FuelConstraint(10, 5000)])
    print("Environment created")
    for epoch in range(1000):
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
