from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py

# Load the Pong environment
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Load the saved A2C model
model = A2C.load("pong_model_801442738", env=vec_env)

# Run the trained model
obs = vec_env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")