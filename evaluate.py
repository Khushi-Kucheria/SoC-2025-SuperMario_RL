import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym.wrappers import GrayScaleObservation, ResizeObservation
import time


def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env, keep_dim=True)
    return env

env = DummyVecEnv([create_env])
env = VecFrameStack(env, n_stack=4)


model = PPO.load("train/ppo_mario/mario_model")

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)

env.close()