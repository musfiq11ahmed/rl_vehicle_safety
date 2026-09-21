import gymnasium as gym
import numpy as np
from gymnasium import spaces

class VehicleSafetyEnv(gym.Env):
    def __init__(self):
        super(VehicleSafetyEnv, self).__init__()

        # Simulation parameters
        self.dt = 0.1            # timestep in seconds
        self.max_steps = 200     # max steps per episode
        self.step_count = 0

        # Vehicle parameters
        self.max_speed = 30.0    # m/s
        self.min_speed = 0.0
        self.max_accel = 3.0     # m/s^2
        self.min_accel = -8.0    # m/s^2 (braking)

        # Safety threshold
        self.ttc_threshold = 1.5  # seconds — danger zone

        # Action space: one continuous value — acceleration
        # Range: -8.0 (hard brake) to 3.0 (accelerate)
        self.action_space = spaces.Box(
            low=np.array([self.min_accel]),
            high=np.array([self.max_accel]),
            dtype=np.float32
        )

        # Observation space: [ego_speed, lead_speed, gap, ttc]
        # self.observation_space = spaces.Box(
            #low=np.array([0.0,  0.0,  0.0,  0.0]),
            #high=np.array([30.0, 30.0, 100.0, 10.0]),
            #dtype=np.float32
        
        self.observation_space = spaces.Box(
            low   = np.array([0.0,  0.0,   0.0,  0.0],  dtype=np.float32),
            high  = np.array([30.0, 30.0, 100.0, 10.0],  dtype=np.float32),
            dtype = np.float32
        )
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize initial conditions
        self.ego_speed  = np.random.uniform(10.0, 25.0)  # ego vehicle speed
        self.lead_speed = np.random.uniform(5.0, 20.0)   # lead vehicle speed
        self.gap        = np.random.uniform(20.0, 60.0)  # distance between vehicles
        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        accel = float(np.clip(action[0], self.min_accel, self.max_accel))

        # Update ego vehicle speed
        self.ego_speed += accel * self.dt
        self.ego_speed  = np.clip(self.ego_speed, self.min_speed, self.max_speed)

        # Lead vehicle moves at constant speed
        self.gap += (self.lead_speed - self.ego_speed) * self.dt
        self.step_count += 1

        # Compute TTC
        ttc = self._compute_ttc()

        # Check termination conditions
        collided    = self.gap <= 0.0
        truncated   = self.step_count >= self.max_steps
        terminated  = collided

        # Compute reward
        reward = self._compute_reward(ttc, collided)

        obs  = self._get_obs()
        info = {"ttc": ttc, "gap": self.gap, "ego_speed": self.ego_speed}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        ttc = self._compute_ttc()
        return np.array([
            self.ego_speed,
            self.lead_speed,
            np.clip(self.gap, 0.0, 100.0),
            np.clip(ttc, 0.0, 10.0)
        ], dtype=np.float32)

    def _compute_ttc(self):
        relative_speed = self.ego_speed - self.lead_speed
        if relative_speed <= 0:
            return 10.0  # vehicles diverging — safe
        return float(np.clip(self.gap / relative_speed, 0.0, 10.0))

        # def _compute_reward(self, ttc, collided):
        #if collided:
            #return -200.0
        #if ttc < self.ttc_threshold:
            #return -50.0
        #if ttc < 4.0:
            #return ttc * 2.0
        #return 8.0
    
    def _compute_reward(self, ttc, collided):
        # Collision
        if collided:
            return -200.0

         # Danger zone — TTC below safety threshold
        if ttc < self.ttc_threshold:
            return -50.0

        # Speed penalty — penalizes driving too slowly        
        speed_penalty = max(0, 15.0 - self.ego_speed) * 0.5

        # Safe approach zone — reward scales with TTC margin
        if ttc < 4.0:
            return (ttc * 2.0) - speed_penalty

        # Fully safe zone — maximum reward minus speed penalty
        return 8.0 - speed_penalty