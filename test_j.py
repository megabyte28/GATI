import numpy as np
import pybullet as p
import time
from stable_baselines3 import PPO
from fuck import GatiHeavyEnv # Ensure this matches your training filename

def run_test():
    # 1. Initialize the environment with GUI enabled
    # The wall is automatically built inside the env.reset() we wrote
    env = GatiHeavyEnv(gui=True)
    
    # 2. Load the trained brain
    # Make sure 'gati_drone_model.zip' exists in your folder
    try:
        model = PPO.load("heavy_stable_drone", env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Run the test loop
    obs, info = env.reset()
    
    for i in range(5000): # Run for 5000 steps
        # Predict the best action (deterministic=True)
        action, _states = model.predict(obs, deterministic=True)
        
        # Apply the action to the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Slow down the simulation for the M4 display (Match 48Hz control freq)
        time.sleep(1/48)
        
        if terminated or truncated:
            print("Target reached or collision occurred. Resetting...")
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    run_test()