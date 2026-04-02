import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from stable_baselines3 import PPO
from gym_pybullet_drones.envs.WallAviary import WallAviary

def test():
    env = WallAviary(gui=True)
    
    model = PPO.load("smart_drone_model_aggressive")
    
    obs, info = env.reset()
    
    for i in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        time.sleep(1/60)
        
        
    env.close()

if __name__ == "__main__":
    test()