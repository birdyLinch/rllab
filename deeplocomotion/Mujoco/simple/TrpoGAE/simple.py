import lasagne.nonlinearities as NL

from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.GANimitation.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

# auto save config
experiment_spec = "100X50X25*2_10000|"
save_policy_every = 25
total_iter = 10000
window=4
max_path_length=5000
batch_size=30000

from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=3)


env = normalize(SimpleHumanoidEnv(window=4))

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
    n_itr=total_iter,
    max_path_length=max_path_length,
    experiment_spec=experiment_spec,
    save_policy_every=save_policy_every,
    batch_size=batch_size,
    discount=0.995,
    gae_lambda=0.98,
)

algo.train(),

