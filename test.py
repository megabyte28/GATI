import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel
# Model load karo
model = PPO.load("hover_10m_model_new")

# GUI on karke dekho
env = HoverAviary(drone_model=DroneModel.CF2X, gui=True)
obs, info = env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

        