import numpy as np
from gym.envs.mujoco import HopperEnv

class HopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        super(HopperEnv, self).__init__(**kwargs)

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        
        alive_bonus = 1.0
        ctrl_cost = 1e-3 * np.square(action).sum()
        forward_reward = (next_obs[:, 0] - obs[:, 0])/self.dt
        reward = forward_reward - ctrl_cost + alive_bonus
        return reward
    
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward_run = (posafter - posbefore) / self.dt
        reward_ctrl = -1e-3 * np.square(a).sum()
        reward = reward_run + reward_ctrl + alive_bonus
        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        ob = self._get_obs()
        info = dict(reward_run=reward_run, 
                    reward_ctrl=reward_ctrl,
                    alive_bonus=alive_bonus,
                    obs_for_re=posbefore.reshape(-1,), 
                    next_obs_for_re=posafter.reshape(-1,))
        
        return ob, reward, done, info