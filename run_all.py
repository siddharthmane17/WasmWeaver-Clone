import os

print(" Step 1: Training PPO Agent")
os.system("python3 -m drl.agents.ppo_agent")


print("\nStep 2: Testing Feature Extractor")
os.system("python3 test_feature_extractor.py")

print("tensorboard --logdir=wasmweaver_tensorboard")
