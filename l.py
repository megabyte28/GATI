import numpy as np
import pybullet as p
from gymnasium import spaces
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class GatiDroneEnv(BaseAviary):
    PYB_FREQ = 240
    CTRL_FREQ = 48

    def __init__(self, gui=True):
        self.INIT_XYZS = np.array([[0, 0, 1]])
        self.INIT_RPYS = np.array([[0, 0, 0]])
        self.TARGET_POS = np.array([2.0, 1.0, 1.0]) # The Side Goal
        self.HOVER_RPM = 14468.0
        self.LIDAR_RANGE = 3.0
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
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos, quat, vel = state[0:3], state[3:7], state[10:13]
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
        forward_vec = rot_mat @ np.array([1, 0, 0])
        ray = p.rayTest(pos, pos + forward_vec * self.LIDAR_RANGE)
        self.current_lidar = ray[0][2] * self.LIDAR_RANGE
        return np.concatenate([[self.current_lidar], self.TARGET_POS - pos, vel]).astype('float32')

    def _preprocessAction(self, action):
        # Extremely tight control for hover stability
        act = np.clip(action, -0.8, 0.8) 
        thrust = self.HOVER_RPM + (act * 1000) # Lower range = smoother flight
        return np.array([thrust]).reshape(1, 4)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos, vel = state[0:3], state[10:13]
        dist = np.linalg.norm(self.TARGET_POS - pos)
        
        # 1. Distance Reward (Breadcrumbs)
        reward = (self.prev_dist - dist) * 100
        self.prev_dist = dist
        
        # 2. THE HOVER BRAKE (Velocity Damping)
        # If close to goal, penalize speed. This forces it to stop and hover.
        if dist < 0.5:
            reward -= np.linalg.norm(vel) * 10 
            if dist < 0.1:
                reward += 10.0 # Reward for "Staying" in the hover zone
        
        # 3. ABSOLUTE WALL PENALTY
        # No more complex math, just a hard "Stay Away" penalty
        if self.current_lidar < 0.6:
            reward -= 20.0 

        # 4. FATAL CRASH
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            reward -= 1000 # Absolute deterrent
            
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        # We only stop if it crashes. We want it to LEARN to hover at the target.
        if state[2] < 0.1 or len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            return True
        return False

    def _computeTruncated(self):
        # Give it plenty of time (8 seconds) to find the goal and settle
        return self.step_counter > 400 

    def _computeInfo(self):
        return {}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0])
        wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0], rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, 
                          baseVisualShapeIndex=wall_vis, basePosition=[1.0, 0.0, 1.0])
        return obs, info

# --- Training Execution ---
if __name__ == "__main__":
    env = GatiDroneEnv(gui=False)
    
    # Use a larger 'batch_size' for M4 stability
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=3e-4, 
                batch_size=128,
                n_steps=2048,
                ent_coef=0.01,
                device="mps")
    
    print("Training Precision Hover Model...")
    model.learn(total_timesteps=400000) # Higher steps for perfection
    model.save("gati_hover_model")