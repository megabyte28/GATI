import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from gym_pybullet_drones.envs.WallAviary import WallAviary

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def make_env(rank: int, seed: int = 0):
    def _init():
        env = WallAviary(gui=False)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    N_ENVS = 8
    TOTAL_TIMESTEPS = 1_000_000
    LOG_DIR = "./logs/"
    SAVE_DIR = "./models/"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    eval_env = Monitor(WallAviary(gui=False))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],
                vf=[256, 256, 128],
            )
        ),
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        device="auto",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 50_000 // N_ENVS),
        save_path=SAVE_DIR,
        name_prefix="drone_nav",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR + "best/",
        log_path=LOG_DIR,
        eval_freq=max(1, 25_000 // N_ENVS),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(SAVE_DIR + "smart_drone_final")

    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    train()