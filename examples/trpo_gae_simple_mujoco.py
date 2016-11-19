import lasagne.nonlinearities as NL

from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from examples.discriminator import Mlp_Discriminator
import pickle

# auto save config
experiment_spec = "100X50X25_simplehumanoid_TRPO_GAE"
save_policy_every = 200
total_iter = 1

from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=3)

try:
    discriminator = Mlp_Discriminator(disc_window=2, iteration=total_iter,disc_joints_dim=56, hidden_sizes=(32, 32))

    # baseline
    env = normalize(SimpleHumanoidEnv())
    # GAN imitaion
    # env = normalize(HumanEnv(discriminator=discriminator))

    print(env.action_space.bounds)
    print(env.observation_space.bounds)

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
        max_path_lenght=2000,
        experiment_spec=experiment_spec,
        save_policy_every=save_policy_every,
        batch_size=5,
        discount=0.995,
        gae_lambda=0.98,

        # baseline
        discriminator=None,

        # GAN imitation
        # discriminator=discriminator,
    )

    algo.train(),

    pickle.dump(policy, open("model/model1.pickle","wb"))
except Exception as e:
    print(e)
