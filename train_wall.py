import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from stable_baselines3 import PPO
from gym_pybullet_drones.envs.WallAviary import WallAviary

def train():
    env = WallAviary(gui=False)

    model = PPO("MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.003,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                device="auto",
                target_kl=0.4,
                ent_coef=0.5)

    start_time = time.time()
    model.learn(total_timesteps=200000)
    end_time = time.time()

    model.save("smart_drone_model_aggressive")
    
    print(f"Total Time: {(end_time - start_time)/60:.2f} mins")
    env.close()

if __name__ == "__main__":
    train()