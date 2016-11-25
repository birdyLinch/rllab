import lasagne.nonlinearities as NL

from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.envs.mujoco.GANimitation.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from deeplocomotion.Mujoco.discriminator_mujoco import Mlp_Discriminator

# config
experiment_spec = "8X8_a(0.02-0.1)_10000|"
save_policy_every = 25
obs_window = 3
total_iter = 10000
max_path_length=5000
batch_size = 30000
Policy="MLP"

from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=10)

discriminator = Mlp_Discriminator( a_max=0.2, a_min=0.2, disc_window=obs_window, iteration=total_iter, disc_joints_dim=2, hidden_sizes=(8, 8))

# initializing
env = normalize(SimpleHumanoidEnv(discriminator=discriminator, window=obs_window), normalize_obs=True)

if Policy=="MLP":
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)
    )

if Policy=="GRU":
    policy=GaussianGRUPolicy(
        env_spec=env.spec,
        hidden_sizes=(64,),
        state_include_action=False,
        hidden_nonlinearity=NL.tanh)

if policy==None:
    print("not valid policy type")
else:
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
            "std_nonlinearity": None,
            "normalize_inputs": True,
            "normalize_outputs": True,
        })

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        n_itr=total_iter,
        max_path_length=max_path_length,
        experiment_spec=experiment_spec,
        save_policy_every=save_policy_every,
        batch_size=batch_size,
        discount=0.995,
        gae_lambda=0.98,
        discriminator=discriminator,
    )

    algo.train()
