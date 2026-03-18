import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel


env = HoverAviary(drone_model=DroneModel.CF2X, gui=False)

try:
    log_path = "./ppo_drone_tensorboard/"
    import tensorboard
except ImportError:
    log_path = None
    print("Warning: Tensorboard nahi mila, bina graph ke train hoga.")

model = PPO("MlpPolicy", 
            env, 
            verbose=1, 
            device="cpu", 
            tensorboard_log=log_path)


print("M4 Beast mode ON! Training starting for 1 Lakh steps...")
model.learn(total_timesteps=500000)


model.save("hover_10m_model_new")
print("Bhai, model save ho gaya. Ab so jao!")