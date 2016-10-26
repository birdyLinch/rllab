from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.HumanEnv import HumanEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(HumanEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)

algo.train()