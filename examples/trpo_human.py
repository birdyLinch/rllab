from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.HumanEnv import HumanEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
import pickle
from subprocess import Popen

Popen(["./HumanDemoNoGUI"])

env = normalize(HumanEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(64, 64)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    n_itr=20000,
)

algo.train(),

pickle.dump(policy, open("model/model1.pickle","wb"))
