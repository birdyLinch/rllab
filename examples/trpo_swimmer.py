from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


env = normalize(SimpleHumanoidEnv())
from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=4)
policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=500,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
algo.train()