import numpy as np
import pybullet as p
from gymnasium import spaces
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class GatiHeavyEnv(BaseAviary):
    # Hardcoded Class Attributes to stop ZeroDivisionError
    PYB_FREQ = 240
    CTRL_FREQ = 48
    IMG_RES = np.array([64, 48])

    def __init__(self, gui=True):
        # 1. 540g Drone Physics Constants
        self.HOVER_RPM = 64745.0  # Calculated for 0.54kg mass
        self.TARGET_POS = np.array([2.5, 1.5, 1.0])
        self.INIT_XYZS = np.array([[0, 0, 1.0]])
        self.INIT_RPYS = np.array([[0, 0, 0]])
        
        # 2. Pre-fill variables for BaseAviary Housekeeping
        self.NUM_DRONES = 1
        self.LIDAR_RANGE = 3.0
        self.prev_dist = np.linalg.norm(self.TARGET_POS - self.INIT_XYZS[0])

        # 3. Initialize BaseAviary
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
        # 4 Motors: [-1, 1] scaled in _preprocessAction
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        # 10 Inputs: Rel_Pos(3), Vel(3), Ang_Vel(3), Lidar(1)
        return spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos, quat, vel, ang_vel = state[0:3], state[3:7], state[10:13], state[13:16]
        
        # LiDAR calculation
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
        forward = rot_mat @ np.array([1, 0, 0])
        ray = p.rayTest(pos, pos + forward * self.LIDAR_RANGE)
        self.current_lidar = ray[0][2] * self.LIDAR_RANGE
        
        return np.concatenate([self.TARGET_POS - pos, vel, ang_vel, [self.current_lidar]]).astype('float32')

    def _preprocessAction(self, action):
        # Heavy drone (540g) needs higher RPM range for stability
        # +/- 10,000 RPM range around the 64.7k base
        act = np.clip(action, -1, 1)
        thrust = self.HOVER_RPM + (act * 10000)
        return np.array([thrust]).reshape(1, 4)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos, vel, ang_vel = state[0:3], state[10:13], state[13:16]
        dist = np.linalg.norm(self.TARGET_POS - pos)
        
        reward = 0
        
        # 1. Z-AXIS STABILITY (Crucial for heavy takeoff)
        z_error = abs(pos[2] - 1.0)
        reward -= z_error * 25 # Penalty for dropping or rising too fast
        
        # 2. PROGRESS
        reward += (self.prev_dist - dist) * 300
        self.prev_dist = dist
        
        # 3. PRECISION HOVER (Stop at target)
        if dist < 0.3:
            reward += 20.0 
            reward -= np.linalg.norm(vel) * 40 # Heavy braking
            reward -= np.linalg.norm(ang_vel) * 10 # Stop rotation

        # 4. WALL REPELLENT
        if self.current_lidar < 0.6:
            reward -= 50.0 

        # 5. CRASH
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            reward -= 5000 
            
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        # Die if it falls or crashes
        if state[2] < 0.1 or len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter > 800 # 16 seconds to reach and hover

    def _computeInfo(self): return {}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        # Create Wall at X=1.0
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0]), 
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0], rgbaColor=[1,0,0,1]),
                          basePosition=[1.0, 0.0, 1.0])
        return obs, info

# --- MAIN ---
if __name__ == "__main__":
    # Train without GUI for speed on M4
    env = GatiHeavyEnv(gui=False)
    
    # Adjusted PPO parameters for heavy drone inertia
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=1e-4, 
                n_steps=4096, 
                batch_size=128, 
                device="mps")
    
    print("Training 540g Heavy Drone on M4...")
    model.learn(total_timesteps=800000) # Increased steps for precision
    model.save("heavy_stable_drone")