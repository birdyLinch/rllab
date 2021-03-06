import lasagne.nonlinearities as NL

from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from examples.HumanEnv_v2 import HumanEnv_v2
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from examples.discriminator_bullet import Mlp_Discriminator
import pickle
from subprocess import Popen
import time
import os
import signal
import sys, traceback

# auto save config
experiment_spec = "100X50X25_22D_DiscriminateReward_GAE"
save_policy_every = 50

from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=1)

simulator =Popen(["./HumanDemoNoGUI"])
time.sleep(3)

# try:
discriminator = Mlp_Discriminator(a_max=0.8, a_min=0.5, decent_portion=0.8, disc_window=2, iteration=3000,disc_joints_dim=16, hidden_sizes=(128, 64, 32))

# baseline
#env = normalize(HumanEnv_v2(discriminator=None), normalize_obs=True)
# GAN imitaion
env = normalize(HumanEnv_v2(discriminator=discriminator), normalize_obs=True)
# print(env.action_space.bounds)
# print(env.observation_space.bounds)

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(100, 50, 25)
)

base_line_optimizer = ConjugateGradientOptimizer()
baseline = GaussianMLPBaseline(env.spec, 
    regressor_args={
        "mean_network": None,
        "hidden_sizes": (100, 50, 25),
        "hidden_nonlinearity": NL.tanh,
        "optimizer": base_line_optimizer,
        "use_trust_region": True,
        "step_size": 0.01,
        "learn_std": True,
        "init_std": 1.0,
        "adaptive_std": False,
        "std_share_network": False,
        "std_hidden_sizes": (32, 32),
        "std_nonlinearity": None,
        "normalize_inputs": True,
        "normalize_outputs": True,
    })

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    n_itr=3000,
    max_path_lenght=2000,
    experiment_spec=experiment_spec,
    save_policy_every=save_policy_every,
    batch_size=30000,
    discount=0.995,
    gae_lambda=0.98,
    step_size=0.01,

    # baseline
    #discriminator=None,

    # GAN imitation
    discriminator=discriminator,
)

algo.train(),

pickle.dump(policy, open("model/model1.pickle","wb"))
# except Exception as e:
#     exc_type, exc_value, exc_traceback = sys.exc_info()
#     traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
#     print(e)

#os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)
