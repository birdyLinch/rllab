import pickle
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from examples.HumanEnv_v2 import HumanEnv_v2
from rllab.envs.normalized_env import normalize
from subprocess import Popen
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import signal

# auto save config
experiment_spec = "100X50X25_22D_PlainReward_GAE"
save_policy_every = 2000
discriminate = False

# show result config
iter_each_policy = 10
max_path_len = 5000

simulator = Popen(["./App_ExampleBrowser"])
try:
    time.sleep(3)
    env = normalize(HumanEnv_v2())

    exper_num = 0

    rewards = []
    all_rewards = []

    while True:
        try:
            itr_str = str((exper_num+1)*save_policy_every)
            
            if discriminate:
                discriminator = pickle.load(open("model/"+experiment_spec+itr_str+"discriminator.pickle","rb"))
                policy = pickle.load(open("model/"+experiment_spec+itr_str+"policy.pickle","rb"))
            else:    
                policy = pickle.load(open("model/"+experiment_spec+itr_str+".pickle","rb"))
            
            exper_num+=1
            tol_reward = 0
            for i in range(iter_each_policy):
                observation = env.reset()
                sum_reward = 0
                for t in range(max_path_len):
                    # show policy
                    action, _ = policy.get_action(observation)
                    # show random
                    # action = env.action_space.sample()
                    observation, reward, done, _ = env.step(action)
                    if done:
                        break
                    sum_reward+= reward
                rewards.append(sum_reward)
            all_rewards.append(rewards)

        except Exception as e:
            print(e)
            break

    all_rewards = np.array(all_rewards)
    
    # plot
    x = np.arange(0, exper_num*save_policy_every, save_policy_every)
    y = np.mean(all_rewards, axis=1)
    yerr = np.std(all_rewards, axis=1)

    plt.errorbar(x, y, xerr=0.0, yerr=yerr)
    plt.show()
except Exception as e:
    print(e)
    pass

os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)
