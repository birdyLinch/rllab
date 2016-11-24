from rllab.envs.base import Step
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import autoargs
from rllab import spaces

BIG = 1e6

class SimpleHumanoidEnv(MujocoEnv, Serializable):

    FILE = 'simple_humanoid_origin.xml'

    @autoargs.arg('vel_deviation_cost_coeff', type=float,
                  help='cost coefficient for velocity deviation')
    @autoargs.arg('alive_bonus', type=float,
                  help='bonus reward for being alive')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for control inputs')
    @autoargs.arg('impact_cost_coeff', type=float,
                  help='cost coefficient for impact')
    def __init__(
            self, velocity_clip=None,
            window=2,
            vel_deviation_cost_coeff=1e-2,
            alive_bonus=0.2,
            ctrl_cost_coeff=1e-5,
            impact_cost_coeff=1e-5,
            discriminator=None,
            *args, **kwargs):
        self.vel_deviation_cost_coeff = vel_deviation_cost_coeff
        self.alive_bonus = alive_bonus
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        self.velocity_clip = velocity_clip
        self.discriminator = discriminator
        self.window=window
        if self.discriminator!=None:
            self.window=self.discriminator.disc_window
        super(SimpleHumanoidEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        
        
    @overrides
    def get_current_obs(self):
        data = self.model.data
        # print(data.cacc.shape)

        return np.concatenate([
            data.qpos.flat,
            # data.qvel.flat,
            # np.clip(data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso").flat,
        ])

    
    @overrides
    @property
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        temp = []
        for i in range(self.window):
            temp.append(ub)
        ub = np.concatenate(temp)
        return spaces.Box(ub * -1, ub)

    def _get_com(self):
        data = self.model.data
        mass = self.model.body_mass
        xpos = data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def step(self, action):
        self.forward_dynamics(action)
        self.state[:self.ob_dim*(self.window-1)] = self.state[self.ob_dim:]
        self.state[self.ob_dim*(self.window-1):] = self.get_current_obs()
        next_obs = self.state

        alive_bonus = self.alive_bonus
        data = self.model.data

        comvel = self.get_body_comvel("torso")

        lin_vel_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        impact_cost = .5 * self.impact_cost_coeff * np.sum(
            np.square(np.clip(data.cfrc_ext, -1, 1)))
        vel_deviation_cost = 0.5 * self.vel_deviation_cost_coeff * np.sum(
            np.square(comvel[1:]))
        if (self.velocity_clip==None or self.velocity_clip<=0):
            reward = lin_vel_reward + alive_bonus - ctrl_cost - \
                impact_cost - vel_deviation_cost
        else:
            reward = lin_vel_reward + alive_bonus - ctrl_cost - \
            impact_cost - vel_deviation_cost
        
        if self.discriminator!=None:
            if lin_vel_reward>=0.1:
                a = self.discriminator.get_a()
                reward = a* self.discriminator.get_reward(self.state) + reward
            self.discriminator.inc_iter()

        done = data.qpos[2] < 0.8 or data.qpos[2] > 2.0

        return Step(next_obs, reward, done)

    @overrides
    def reset(self, init_state=None):
        self.ob_dim=self.observation_space.shape[0]/self.window
        self.state = np.zeros(self.ob_dim*self.window)
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        self.state[self.ob_dim*(self.window-1):] = self.get_current_obs()
        return self.state

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
