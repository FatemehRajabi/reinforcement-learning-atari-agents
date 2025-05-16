# UNC Charlotte
# ITCS 5153 - Applied AI - Spring 2025
# Lab 5
# Reinforcement Learning
# This module trains and tests a PPO model to play MsPacman using Stable-Baselines3.
# Student ID: 801442738

# a. Environment: Ms Pacman (MsPacmanNoFrameskip-v4)
# b. Algorithm: PPO (Proximal Policy Optimization)
# c. Configuration Parameters:
#    - Policy: CnnPolicy
#    - Vectorized Environments: 4 (n_envs=4)
#    - Frame Stack: 4 frames
#    - Total Timesteps: 1,000,000 (1007616)
#    - Learning Rate: 0.0003 (default)
#    - Seed: 0
# d. Training Duration: 2 hours and 30 minutes
# e. GPU Used: No (model trained on CPU only)
# f. System Specs:
#    - RAM: 16.0 GB (4267 MHz)
#    - GPU: Intel(R) Iris(R) Xe Graphics – 2 GB (not used for training)
# g. OS: Windows 11 Home (64-bit)
# h. CPU: 13th Gen Intel(R) Core(TM) i7-1355U @ 1.70 GHz

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import time

import ale_py

# Environment setup
vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Model setup
model = PPO("CnnPolicy", vec_env, verbose=1)

# Measure training time
start_time = time.time()

# Train the PPO model for 1,000,000 timesteps
model.learn(total_timesteps=1_000_000)

end_time = time.time()

# Print training time
elapsed_time = end_time - start_time
print(f"Training time: {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds")

# Save model
model.save("mspacman_model_801442738")

# Test the trained model by running it in the environment
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
