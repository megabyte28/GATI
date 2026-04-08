import numpy as np
import pybullet as p
from gymnasium import spaces
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class GatiDroneEnv(BaseAviary):
    # FORCE these values at the class level to prevent ZeroDivisionError
    PYB_FREQ = 240
    CTRL_FREQ = 48
    IMG_RES = np.array([64, 48])

    def __init__(self, gui=True):
        # 1. Setup initial positions
        self.INIT_XYZS = np.array([[0, 0, 1]])
        self.INIT_RPYS = np.array([[0, 0, 0]])
        self.TARGET_POS = np.array([2.0, 1.0, 1.0]) # Goal is to the side
        self.HOVER_RPM = 14468.0
        self.LIDAR_RANGE = 3.0
        self.prev_dist = np.linalg.norm(self.TARGET_POS - self.INIT_XYZS[0])

        # 2. Initialize BaseAviary with basic positional arguments
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
        # 4 Motors: RPM commands scaled from -1 to 1
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        # 7 Inputs: [Lidar, Rel_Goal_X, Y, Z, Vel_X, Y, Z]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def _computeObs(self):
        # The "Eyes": Standard state + 1D LiDAR
        state = self._getDroneStateVector(0)
        pos, quat, vel = state[0:3], state[3:7], state[10:13]
        
        # Calculate where the "Nose" is pointing
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
        forward_vec = rot_mat @ np.array([1, 0, 0])
        
        # Ray cast (The Yellow Line)
        ray = p.rayTest(pos, pos + forward_vec * self.LIDAR_RANGE)
        self.current_lidar = ray[0][2] * self.LIDAR_RANGE
        
        return np.concatenate([[self.current_lidar], self.TARGET_POS - pos, vel]).astype('float32')

    def _preprocessAction(self, action):
        # The "Muscles": Convert AI output to Motor RPM
        # We ensure it stays near hover so it doesn't just fall
        return np.array([self.HOVER_RPM * (1 + 0.15 * action)]).reshape(1, 4)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        
        # Reward 1: Progress (Positive)
        reward = (self.prev_dist - dist) * 100
        self.prev_dist = dist
        
        # Reward 2: Wall Repellent (Negative)
        if self.current_lidar < 0.5:
            reward -= 5.0
            
        # Reward 3: Collision (Big Negative)
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            reward -= 200
            
        # Reward 4: Success (Big Positive)
        if dist < 0.2:
            reward += 1000
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        # End if it hits floor or goal
        if state[2] < 0.1 or np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.2:
            return True
        # End if it hits the wall
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0])) > 0:
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter > 500

    def _computeInfo(self):
        return {}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        # Re-build the wall every reset so it doesn't disappear
        wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0])
        wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 1.0], rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, 
                          baseVisualShapeIndex=wall_vis, basePosition=[1.0, 0.0, 1.0])
        return obs, info

# --- Main Execution ---
if __name__ == "__main__":
    env = GatiDroneEnv(gui=False)
    
    # Using PPO for training
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)
    
    print("M4 Training Starting... The wall is red, the goal is at [2, 1, 1].")
    model.learn(total_timesteps=100000)
    model.save("gati_drone_model")