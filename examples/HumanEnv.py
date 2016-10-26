from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np

import socket
import struct
import logging

logger = logging.getLogger(__name__)

class HumanEnv(Env):

    def __init__(self):
        # Connect to the simulation in Bullet
        HOST, PORT = 'localhost', 47138
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))
        self.s.send(b'Hello!')

        # Height at which to fail the episode
        self.y_threshold = 0.5


    @property
    def observation_space(self):
        return Box(low=-1e6, high=1e6, shape=(306,))

    @property
    def action_space(self):
        return Box(low=-0.5, high=0.5, shape=(16,))

    def reset(self):
        # Reset the simulation
        self.s.send(b'RESET')
        zeros = np.zeros(20).astype(int)
        buff = struct.pack('%si' % len(zeros), *zeros)
        self.s.send(buff)

        # Get the state
        stateSize = self.s.recv(4);
        stateSize = struct.unpack('i',stateSize)[0]
        state = self.s.recv(1000)
        while len(state) < stateSize:
            state += self.s.recv(1000)
        state = np.asarray([struct.unpack('f',state[i:i+4])[0] for i in range(0,len(state),4)])
        self.state = state
        # # Apply a random control to just have a new initialization every time
        # c = np.random.normal(0.0, 0.2, 16)
        # c = np.clip(c, -1.5, 1.5)
        # buff = struct.pack('%sf' % len(c), *c)
        # self.s.send(buff)

        # # Get the state
        # stateSize = self.s.recv(4);
        # stateSize = struct.unpack('i',stateSize)[0]
        # state = self.s.recv(1000)
        # while len(state) < stateSize:
        #     state += self.s.recv(1000)
        # self.state = np.asarray([struct.unpack('f',state[i:i+4])[0] for i in range(0,len(state),4)])

        self.steps_beyond_done = None
        return np.array(self.state)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        
        # Apply the action
        c = np.asarray(action)
        buff = struct.pack('%sf' % len(c), *c)
        self.s.send(buff)

        # Receive the state
        stateSize = self.s.recv(4);
        stateSize = struct.unpack('i',stateSize)[0]
        state = self.s.recv(1000)
        while len(state) < stateSize:
            state += self.s.recv(1000)
        state = np.asarray([struct.unpack('f',state[i:i+4])[0] for i in range(0,len(state),4)])

        self.state = state

        # Get the y position of the root joint
        y = state[3]
        done = y < self.y_threshold

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # skeleton just fell!
            self.steps_beyond_done = 0
            reward = -100.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        next_observation = np.array(self.state)
        self.state = next_observation
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print (self.state[0:4])
