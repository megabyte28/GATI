import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.SmartDroneEnv import SmartDroneEnv

def start_training():
    # M4 Pe training tez karne ke liye gui=False
    env = SmartDroneEnv(gui=False)
    
    # PPO Algorithm: Sabse best drones ke liye
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("M4 Power Starting... Drone seekhna shuru kar raha hai!")
    model.learn(total_timesteps=300000) # 3 Lakh steps for better results
    
    model.save("wall")
    print("Training Khatam! Model saved.")

if __name__ == "__main__":
    start_training()