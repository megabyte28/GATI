import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class WallAviary(BaseRLAviary):
    NUM_LIDAR_RAYS = 36
    LIDAR_MAX_RANGE = 5.0
    SONAR_MAX_RANGE = 3.0
    OBS_DIM = 12 + 3 + NUM_LIDAR_RAYS + 2

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
    ):
        self.TARGET_POS = np.array([2.0, 0.0, 1.0])
        self.OBSTACLE_POS = np.array([1.0, 0.0, 1.0])
        self.EPISODE_LEN_SEC = 12
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs if initial_xyzs is not None else np.array([[0, 0, 0.5]]),
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )
        init_pos = self.INIT_XYZS[0] if hasattr(self, 'INIT_XYZS') else np.array([0, 0, 0.5])
        self.prev_dist = np.linalg.norm(self.TARGET_POS - init_pos)
        self.collision_occurred = False

    def _addObstacles(self):
        wall_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.05, 0.8, 0.8],
            physicsClientId=self.CLIENT
        )
        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05, 0.8, 0.8],
            rgbaColor=[0.8, 0.2, 0.2, 1.0],
            physicsClientId=self.CLIENT
        )
        self.wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=self.OBSTACLE_POS,
            physicsClientId=self.CLIENT
        )

    def _get_horizontal_lidar(self, drone_idx: int) -> np.ndarray:
        pos = self.getDroneStateVector(drone_idx)[0:3]
        lidar_z = pos[2]
        ray_from = []
        ray_to = []
        for i in range(self.NUM_LIDAR_RAYS):
            angle = (2.0 * np.pi * i) / self.NUM_LIDAR_RAYS
            dx = self.LIDAR_MAX_RANGE * np.cos(angle)
            dy = self.LIDAR_MAX_RANGE * np.sin(angle)
            ray_from.append([pos[0], pos[1], lidar_z])
            ray_to.append([pos[0] + dx, pos[1] + dy, lidar_z])
        results = p.rayTestBatch(
            ray_from,
            ray_to,
            physicsClientId=self.CLIENT
        )
        distances = np.array([res[2] for res in results], dtype=np.float32)
        return distances

    def _get_vertical_sonar(self, drone_idx: int) -> np.ndarray:
        pos = self.getDroneStateVector(drone_idx)[0:3]
        ray_down_to = [pos[0], pos[1], pos[2] - self.SONAR_MAX_RANGE]
        ray_up_to = [pos[0], pos[1], pos[2] + self.SONAR_MAX_RANGE]
        results = p.rayTestBatch(
            [pos, pos],
            [ray_down_to, ray_up_to],
            physicsClientId=self.CLIENT
        )
        down_frac = results[0][2]
        up_frac = results[1][2]
        return np.array([down_frac, up_frac], dtype=np.float32)

    def _check_collision(self, drone_idx: int) -> bool:
        contacts = p.getContactPoints(
            bodyA=self.DRONE_IDS[drone_idx],
            bodyB=self.wall_id,
            physicsClientId=self.CLIENT
        )
        return len(contacts) > 0

    def _observationSpace(self):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32
        )

    def _computeObs(self):
        state = self.getDroneStateVector(0)
        kin = np.hstack([
            state[0:3],
            state[7:10],
            state[10:13],
            state[13:16],
        ])
        rel_target = (self.TARGET_POS - state[0:3]) / 4.0
        lidar_2d = self._get_horizontal_lidar(0)
        sonar_1d = self._get_vertical_sonar(0)
        obs = np.hstack([kin, rel_target, lidar_2d, sonar_1d]).astype(np.float32)
        assert obs.shape == (self.OBS_DIM,), f"Obs shape mismatch: {obs.shape}"
        return obs

    def _computeReward(self):
        state = self.getDroneStateVector(0)
        pos = state[0:3]
        dist_to_target = np.linalg.norm(self.TARGET_POS - pos)
        progress = self.prev_dist - dist_to_target
        reward_progress = progress * 50.0
        reward_goal = 0.0
        if dist_to_target < 0.15:
            reward_goal = +200.0
        lidar = self._get_horizontal_lidar(0)
        sonar = self._get_vertical_sonar(0)
        min_lidar = np.min(lidar)
        actual_min_dist = min_lidar * self.LIDAR_MAX_RANGE
        ALPHA = 5.0
        BETA = 3.0
        reward_proximity = -ALPHA * np.exp(-BETA * actual_min_dist)
        self.collision_occurred = self._check_collision(0)
        reward_collision = -150.0 if self.collision_occurred else 0.0
        target_alt = self.TARGET_POS[2]
        alt_error = abs(pos[2] - target_alt)
        reward_altitude = -2.0 * (alt_error ** 2)
        rpy = state[7:10]
        tilt = np.sqrt(rpy[0]**2 + rpy[1]**2)
        reward_attitude = -1.0 * max(0.0, tilt - 0.3)
        reward_time = -0.1
        reward = (
            reward_progress
            + reward_goal
            + reward_proximity
            + reward_collision
            + reward_altitude
            + reward_attitude
            + reward_time
        )
        self.prev_dist = dist_to_target
        return float(reward)

    def _computeTerminated(self):
        state = self.getDroneStateVector(0)
        pos = state[0:3]
        if np.linalg.norm(self.TARGET_POS - pos) < 0.15:
            return True
        if self.collision_occurred:
            return True
        return False

    def _computeTruncated(self):
        state = self.getDroneStateVector(0)
        pos = state[0:3]
        if abs(pos[0]) > 4.0 or abs(pos[1]) > 4.0:
            return True
        if pos[2] < 0.05 or pos[2] > 4.0:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self.getDroneStateVector(0)
        return {
            "dist_to_target": float(np.linalg.norm(self.TARGET_POS - state[0:3])),
            "collision": self.collision_occurred,
        }
