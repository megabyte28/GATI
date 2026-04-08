import time
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.WallAviary import WallAviary 

def test_drone():
    env = WallAviary(gui=True, record=False) 
    model_path = "ppo_wall_drone" 
    
    try:
        model = PPO.load("models/best/best_model.zip", env=env)
        print(f"Model '{model_path}' loaded successfully! 🚀")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    obs, info = env.reset()
    print("Testing started. Press CTRL+C in terminal to stop.")
    
    try:
        for i in range(10000): 
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(1./240.) 
            
            if terminated or truncated:
                print(f"Episode Finished! Reward: {reward}")
                time.sleep(1) 
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nTesting stopped by user.")
    
    finally:
        env.close()

if __name__ == "__main__":
    test_drone()