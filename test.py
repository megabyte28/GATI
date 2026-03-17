import gym
import gym_pybullet_drones
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import numpy as np
env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
obs = env.reset()
for i in range(1000):
    action = np.array([[15000,15000,15000,15000]]) 
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()