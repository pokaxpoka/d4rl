import numpy as np
from gym.envs.mujoco import HalfCheetahEnv

class HalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        super(HalfCheetahEnv, self).__init__(**kwargs)
    
    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        ctrl_cost = 0.1 * np.sum(np.square(action), axis=1)
        forward_reward = (next_obs[:, 0] - obs[:, 0])/self.dt
        reward = forward_reward - ctrl_cost
        return reward
    
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        info = dict(reward_run=reward_run, 
                    reward_ctrl=reward_ctrl, 
                    obs_for_re=xposbefore.reshape(-1,), 
                    next_obs_for_re=xposafter.reshape(-1,))
        return ob, reward, done, info