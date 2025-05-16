from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py

# Set up the environment
vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Load the saved model
model = PPO.load("mspacman_model_801442738", env=vec_env)

# Run the model in the environment
obs = vec_env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
