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
            decent_portion=0.8,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=NL.tanh,
            disc_network=None,
    ):
        self.disc_window = disc_window
        self.disc_joints_dim = disc_joints_dim
        self.disc_dim = self.disc_window*self.disc_joints_dim
        self.end_iter = int(iteration*decent_portion)
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

    def get_reward(self, disc_obs):
        flat_obs = np.asarray(disc_obs).flatten()
        assert(flat_obs.shape[0] == self.disc_dim)
        reward = self._f_disc([flat_obs])
        return reward

    def train(self, disc_obs, targets):
        logger.log("fitting discriminator...")
        flat_obs = self.disc_space.flatten(disc_obs)
        loss = self._f_disc_train(flat_obs, targets)
        logger.record_tabular("DiscriminatorLoss", loss)

    def load_data(self, fileName='MocapData.mat'):
        # TODO: X (n, dim) dim must equals to the disc_obs
        data=sio.loadmat(fileName)['data'][0]
        X = np.concatenate([np.asarray(frame) for frame in data],0)
        usedDim = np.ones(X.shape[1]).astype('bool')
        usedDim[39:45] = False
        X = X[:,usedDim]
        assert(X.shape[1] == self.disc_joints_dim)
        return X

    def get_batch(self, batch_size):
        mask = np.random.randint(0, self.data.shape[0]-self.disc_window, size=batch_size)
        tensor_list =[]
        for i in range(self.disc_window):
            tensor_list.append(self.data[mask+i])
        return np.hstack(tensor_list)

    def inc_iter(self):
        self.iteration+=1

    def get_a(self):
        if self.iteration < self.end_iter:
            return self.a[self.iteration]
        else:
            return self.a[-1]
