# UNC Charlotte
# ITCS 5153 - Applied AI - Spring 2025
# Lab 5
# Reinforcement Learning
# This module trains and tests an A2C model to play Pong using Stable-Baselines3.
# Student ID: 801442738

# a. Environment: Pong (PongNoFrameskip-v4)
# b. Algorithm: A2C (Advantage Actor-Critic)
# c. Configuration Parameters:
#    - Policy: CnnPolicy
#    - Vectorized Environments: 4 (n_envs=4)
#    - Frame Stack: 4 frames
#    - Total Timesteps: 2,000,000
#    - Learning Rate: 0.0007 (default)
#    - Seed: 0
# d. Training Duration: 2 hours and 29 minutes
# e. GPU Used: No (model trained on CPU only)
# f. System Specs:
#    - RAM: 16.0 GB (4267 MHz)
#    - GPU: Intel(R) Iris(R) Xe Graphics – 2 GB (not used for training)
# g. OS: Windows 11 Home (64-bit)
# h. CPU: 13th Gen Intel(R) Core(TM) i7-1355U @ 1.70 GHz

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C
import time

import ale_py

# Environment setup
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Model setup
model = A2C("CnnPolicy", vec_env, verbose=1)

# Measure training time
start_time = time.time()

# Train the A2C model for 2,000,000 timesteps
model.learn(total_timesteps=2_000_000)

end_time = time.time()

# Print training time
elapsed_time = end_time - start_time
print(f"Training time: {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds")

# Save model
model.save("pong_model_801442738")

# Test the trained model by running it in the environment
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")