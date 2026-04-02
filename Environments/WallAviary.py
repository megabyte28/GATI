import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class WallAviary(BaseRLAviary):

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        
        self.TARGET_POS = np.array([2.0, 0.0, 1.0])
        self.OBSTACLE_POS = np.array([1.0, 0.0, 1.0]) 
        self.EPISODE_LEN_SEC = 8

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.prev_dist = np.linalg.norm(self.TARGET_POS - self.initial_xyzs[0] if self.initial_xyzs else np.array([0,0,0]))

    def _addObstacles(self):
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 1.0, 1.0])
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 1.0, 1.0], rgbaColor=[0.7, 0.2, 0.2, 1])
        
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=wall_shape,
                          baseVisualShapeIndex=wall_visual,
                          basePosition=self.OBSTACLE_POS, 
                          physicsClientId=self.CLIENT)
    def get_1d_downward_lidar(self,drone_index,max_range=3.0):
        pos = self._getDroneStateVector(drone_index)[0:3]
        ray_to = (pos[0],pos[1],pos[2]-max_range)
        result = p.rayTest(pos,ray_to,physicsClientId=self.CLIENT[0])
        return np.array([result[2]])
    def get_2d_lidar_data(self, drone_index, num_rays=360, ray_length=8.0):
            pos = self._getDroneStateVector(drone_index)[0:3]
            lidar_z = pos[2] + 0.05 
            ray_from = []
            ray_to = []
            for i in range(num_rays):
                angle = (2 * np.pi * i) / num_rays 
                ray_from.append([pos[0], pos[1], lidar_z])
                ray_to.append([pos[0] + ray_length * np.cos(angle), pos[1] + ray_length * np.sin(angle), lidar_z])
            results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.CLIENT)
            return np.array([res[2] for res in results])
    def _observationSpace(self):
            lo = -np.inf
            hi = np.inf
            return spaces.Box(low=lo, high=hi, shape=(self.NUM_DRONES, 376), dtype=np.float32)
    def _computeObs(self):
            obs_376 = np.zeros((self.NUM_DRONES, 376))
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                kin_12 = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]])
                rel_target = self.TARGET_POS - state[0:3]
                dist_1d = self.get_1d_downward_lidar(i) 
                dist_2d = self.get_2d_lidar_data(i)     
                obs_376[i, :] = np.hstack([kin_12, rel_target, dist_1d, dist_2d])
            return obs_376.astype('float32')
    def _computeReward(self):
            state = self._getDroneStateVector(0)
            pos = state[0:3]
            reward = 0.0
            dist_to_target = np.linalg.norm(self.TARGET_POS - pos)
            progress = self.prev_dist - dist_to_target
            reward += progress * 50.0  
            dist_2d_array = self.get_2d_lidar_data(0) * 8.0 
            min_2d_dist = np.min(dist_2d_array)
            if min_2d_dist < 0.5:
                reward -= 20.0 * (0.5 - min_2d_dist)
            dist_1d = self.get_1d_downward_lidar(0)[0] * 3.0
            if dist_1d < 0.2:
                reward -= 10.0    
            self.prev_dist = dist_to_target
            return float(reward)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.15:
            return True
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 3.0 or abs(state[1]) > 3.0 or state[2] > 3.0:
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"answer": 42}