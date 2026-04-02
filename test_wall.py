import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.SmartDroneEnv import SmartDroneEnv

model = PPO.load("smart_drone_model2")
env = SmartDroneEnv(gui=True) # Ab maza dekho!

obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1/240)
    #f terminated or truncated:
        #obs, info = env.reset()