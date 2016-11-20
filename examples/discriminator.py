import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init
import theano.tensor as TT

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.misc import ext
from rllab.misc import logger

import scipy.io as sio
import numpy as np

class Mlp_Discriminator(LasagnePowered):
    def __init__(
            self,
            disc_window,
            disc_joints_dim,
            iteration,
            a_max=0.7,
            a_min=0.0,
            batch_size = 64,
            iter_per_train = 10,
            decent_portion=0.8,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=NL.tanh,
            disc_network=None,
    ):  
        self.batch_size=64
        self.iter_per_train=10
        self.disc_window = disc_window
        self.disc_joints_dim = disc_joints_dim
        self.disc_dim = self.disc_window*self.disc_joints_dim
        self.end_iter = int(iteration*decent_portion)
        self.iter_count = 0
        out_dim = 1
        target_var = TT.ivector('targets')

        # create network
        if disc_network is None:
            disc_network = MLP(
                input_shape=(self.disc_dim,),
                output_dim=out_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
            )

        self._disc_network = disc_network

        disc_reward = disc_network.output_layer
        obs_var = disc_network.input_layer.input_var

        disc_var, = L.get_output([disc_reward])

        self._disc_var = disc_var

        LasagnePowered.__init__(self, [disc_reward])
        self._f_disc = ext.compile_function(
            inputs=[obs_var],
            outputs=[disc_var],
            log_name="f_discriminate_forward",
        )
        
        params = L.get_all_params(disc_network, trainable=True)
        loss = lasagne.objectives.categorical_crossentropy(disc_var, target_var).mean()
        updates = lasagne.updates.adam(loss, params, learning_rate=0.01)
        self._f_disc_train = ext.compile_function(
            inputs=[obs_var, target_var],
            outputs=[loss],
            updates=updates,
            log_name="f_discriminate_train"
        )

        self.data = self.load_data()
        self.a = np.linspace(a_min, a_max, self.end_iter)

    def get_reward(self, observation):
        if(len(observation.shape)==1):
            observation = observation.reshape((1, observation.shape[0]))
        disc_ob = self.get_disc_obs(observation)
        assert(disc_ob.shape[1] == self.disc_dim)
        reward = self._f_disc(disc_ob)[0]     
        return reward[0][0]

    def train(self, observations):
        '''
        observations: length trj_num list of np.array with shape (trj_length, dim)
        '''
        logger.log("fitting discriminator...")
        loss=[]
        for i in range(self.iter_per_train):
            batch_obs = self.get_batch_obs(observations, self.batch_size)
            batch_mocap = self.get_batch_mocap(self.batch_size)
            disc_obs = self.get_disc_obs(batch_obs)
            disc_mocap = self.get_disc_mocap(batch_mocap)
            X = np.vstack((disc_obs, disc_mocap))
            targets = np.zeros([2*self.batch_size, 1])
            targets[self.batch_size :]=1
            loss.append(self._f_disc_train(X, targets))
        logger.record_tabular("averageDiscriminatorLoss", np.mean(loss))

    def load_data(self, fileName='MocapData.mat'):
        # TODO: X (n, dim) dim must equals to the disc_obs
        data=sio.loadmat(fileName)['data'][0]
        X = np.concatenate([np.asarray(frame) for frame in data],0)
        usedDim = np.ones(X.shape[1]).astype('bool')
        usedDim[39:45] = False
        X = X[:,usedDim]
        assert(X.shape[1] == self.disc_joints_dim)
        return X

    def get_batch_mocap(self, batch_size):
        '''
        return np.array of shape (batch_size, mocap_dim*window)
        '''
        mask = np.random.randint(0, self.data.shape[0]-self.disc_window, size=batch_size)
        temp =[]
        for i in range(self.disc_window):
            temp.append(self.data[mask+i])
        return np.hstack(temp)

    def get_disc_mocap(self, mocap_batch):
        '''
        param mocap_batch np.array of shape (batch_size, mocap_dim*window)
        return np.array of ashape (batch_size, disc_dim)
        '''
        pass

    def inc_iter(self):
        self.iter_count+=1

    def get_a(self):
        if self.iter_count < self.end_iter:
            return self.a[self.iter_count]
        else:
            return self.a[-1]

    def get_batch_obs(self, observations, batch_size):
        '''
        params observations: length trj_num list of np.array with shape (trj_length, dim)
        params batch_size: batch_size of obs
        return a np.array with shape (batch_size, observation_dim)
        '''
        observations = np.vstack(observations)
        mask = np.random.randint(0, observations.shape[0]-self.window, size=batch_size)
        temp = []
        for i in range(self.disc_window):
            temp.append(self.data[mask+i])
        return np.hstack(temp)


    def get_disc_obs(self, observation):
        """
        param observation nparray with shape (n, window*obs_dim)
        return observation nparray with shape(n, disc_dim)
        """
        pass
