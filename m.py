import numpy as np
import pybullet as p
from gymnasium import spaces
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class GatiFinalEnv(BaseAviary):
    PYB_FREQ, CTRL_FREQ = 240, 48

    def __init__(self, gui=True):
        self.INIT_XYZS = np.array([[0, 0, 1]])
        self.TARGET_POS = np.array([2.5, 1.5, 1.0]) # Clear of the wall
        self.HOVER_RPM = 14468.0
        self.prev_dist = np.linalg.norm(self.TARGET_POS - self.INIT_XYZS[0])
        
        super().__init__(drone_model=DroneModel.CF2X, num_drones=1, 
                         initial_xyzs=self.INIT_XYZS, physics=Physics.PYB, 
                         pyb_freq=self.PYB_FREQ, ctrl_freq=self.CTRL_FREQ, gui=gui)

    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        # [Rel_Pos(3), Vel(3), Ang_Vel(3), Lidar(1)] = 10 inputs
        return spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos, quat, vel, ang_vel = state[0:3], state[3:7], state[10:13], state[13:16]
        
        # LiDAR
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
        forward = rot_mat @ np.array([1, 0, 0])
        ray = p.rayTest(pos, pos + forward * 3.0)
        self.lidar = ray[0][2] * 3.0
        
        return np.concatenate([self.TARGET_POS - pos, vel, ang_vel, [self.lidar]]).astype('float32')

    def _preprocessAction(self, action):
        # Precise thrust mapping
        return np.array([self.HOVER_RPM + (np.clip(action, -1, 1) * 1200)]).reshape(1, 4)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos, vel = state[0:3], state[10:13]
        dist = np.linalg.norm(self.TARGET_POS - pos)
        
        # 1. Distance Progress
        reward = (self.prev_dist - dist) * 200
        self.prev_dist = dist
        
        # 2. THE "FORBIDDEN ZONE" (X between 0.8 and 1.2, Y < 0.8)
        # This is the space the wall occupies. If it enters, it dies.
        if 0.7 < pos[0] < 1.3 and pos[1] < 0.7:
            reward -= 2000 # Massive scolding
            
        # 3. HOVER STABILITY
        if dist < 0.3:
            # Penalize any movement once at target
            reward += 15.0 # "Good drone for staying here"
            reward -= np.linalg.norm(vel) * 20 
        
        # 4. COLLISION
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            reward -= 5000 
            
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        # Die if crash or Forbidden Zone
        if state[2] < 0.1 or len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            return True
        if 0.7 < state[0] < 1.3 and state[1] < 0.7:
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter > 500

    def _computeInfo(self): return {}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0]), 
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0], rgbaColor=[1,0,0,1]),
                          basePosition=[1.0, 0.0, 1.0])
        return obs, info

if __name__ == "__main__":
    env = GatiFinalEnv(gui=False)
    # n_steps=4096 gives the M4 more data per update to stabilize flight
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=4096, batch_size=128, device="mps")
    
    print("Training Final Version... Target is (2.5, 1.5, 1.0). Wall is at (1, 0, 1).")
    model.learn(total_timesteps=600000)
    model.save("gati_modi")