import numpy as np
import pybullet as p
from gymnasium import spaces
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class GatiDroneEnv(BaseAviary):
    # Fixed frequencies for M4 stability
    PYB_FREQ = 240
    CTRL_FREQ = 48

    def __init__(self, gui=True):
        self.INIT_XYZS = np.array([[0, 0, 1]])
        self.INIT_RPYS = np.array([[0, 0, 0]])
        self.TARGET_POS = np.array([2.5, 1.2, 1.0]) # Further side goal for better curves
        self.HOVER_RPM = 14468.0
        self.LIDAR_RANGE = 3.5
        self.prev_dist = np.linalg.norm(self.TARGET_POS - self.INIT_XYZS[0])

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self.INIT_XYZS,
            initial_rpys=self.INIT_RPYS,
            physics=Physics.PYB,
            pyb_freq=self.PYB_FREQ,
            ctrl_freq=self.CTRL_FREQ,
            gui=gui
        )

    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        # [Lidar, Rel_Goal(3), Velocity(3)]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos, quat, vel = state[0:3], state[3:7], state[10:13]
        
        # Ray cast (LiDAR)
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
        forward_vec = rot_mat @ np.array([1, 0, 0])
        ray = p.rayTest(pos, pos + forward_vec * self.LIDAR_RANGE)
        self.current_lidar = ray[0][2] * self.LIDAR_RANGE
        
        return np.concatenate([[self.current_lidar], self.TARGET_POS - pos, vel]).astype('float32')

    def _preprocessAction(self, action):
        # Smoothing: We limit the AI's "jerky" inputs to +/- 10% of hover
        act = np.clip(action, -1, 1)
        thrust = self.HOVER_RPM + (act * 1500) 
        return np.array([thrust]).reshape(1, 4)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos, vel = state[0:3], state[10:13]
        dist = np.linalg.norm(self.TARGET_POS - pos)
        
        # 1. THE PROGRESS (Strong Pull)
        reward = (self.prev_dist - dist) * 150
        self.prev_dist = dist
        
        # 2. THE FORCE FIELD (Exponential Wall Repellent)
        # If the drone gets within 0.8m, the penalty grows exponentially
        if self.current_lidar < 0.8:
            reward -= 2.0 / (self.current_lidar + 0.05)**2 

        # 3. THE SIDE-BIAS (Hinting the Curve)
        # If the drone is behind the wall (X < 1.0) and moving towards the gap (Y > 0)
        if pos[0] < 1.0 and vel[1] > 0.1:
            reward += 5.0 

        # 4. COLLISION & SUCCESS
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            reward -= 500 # "Ouch"
        if dist < 0.2:
            reward += 2000 # "Victory"
            
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if state[2] < 0.1 or np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.2:
            return True
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter > 600

    def _computeInfo(self):
        return {}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        # Re-build the thin wall at X=1.0
        wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0])
        wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0], rgbaColor=[0.8, 0.1, 0.1, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, 
                          baseVisualShapeIndex=wall_vis, basePosition=[1.0, 0.0, 1.0])
        return obs, info

# --- Training Execution ---
if __name__ == "__main__":
    # Training on M4 (MPS)
    env = GatiDroneEnv(gui=False) # GUI off = 10x faster training
    
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=2e-4, # Slower rate for smoother flight
                ent_coef=0.03,      # High entropy to find the "curve"
                device="mps")       # M4 Acceleration
    
    print("Training 'Smooth-Curve' Drone on M4 Neural Engine...")
    model.learn(total_timesteps=250000)
    model.save("gati_smooth_dodger")