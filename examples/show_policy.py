import pickle
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from examples.HumanEnv import HumanEnv
from rllab.envs.normalized_env import normalize
from subprocess import Popen

Popen(["App_ExampleBrowser"])
policy = pickle.load(open("model/model1.pickle","rb"))
env = normalize(HumanEnv())


for i in range(100):
    observation = env.reset()
    for t in range(200):

        # show policy
        action, _ = policy.get_action(observation)
        # show random
        # action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        if done:
            print(reward)
            break