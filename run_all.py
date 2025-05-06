import os

print(" Step 1: Training PPO Agent")
os.system("python3 -m drl.agents.pp_agent")


print("\nStep 2: Testing Feature Extractor")
os.system("python3 test_feature_extractor.py")

print("\n Step 3: Launching TensorBoard (optional)")
print("To run TensorBoard manually, use:")
print("tensorboard --logdir=wasmweaver_tensorboard")
