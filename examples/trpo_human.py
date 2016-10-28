from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.HumanEnv import HumanEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
import pickle
from subprocess import Popen
import time
import os
import signal

experiment_spec = "64X2_26D_NaiveReward"
save_policy_every = 2

simulator =Popen(["./HumanDemoNoGUI"])
time.sleep(3)

try:
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
        n_itr=10,
        max_path_lenght=5000,
        experiment_spec=experiment_spec,
        save_policy_every=save_policy_every,
    )

    algo.train(),

    pickle.dump(policy, open("model/model1.pickle","wb"))
except Exception as e:
    pass

os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)
