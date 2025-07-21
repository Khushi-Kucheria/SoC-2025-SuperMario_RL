import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from gym.wrappers import GrayScaleObservation, ResizeObservation

import os

def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env, keep_dim=True)
    env = Monitor(env)  # log stats like episode rewards
    return env

env = DummyVecEnv([create_env])
env = VecFrameStack(env, n_stack=4)

save_path = os.path.join("train", "ppo_mario")
os.makedirs(save_path, exist_ok=True)

# Create the PPO model
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_mario_logs/")

model.learn(total_timesteps=100000)
model.save(os.path.join(save_path, "mario_model"))

print("Training complete and model saved.")

