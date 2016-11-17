import pickle
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.test_humanoid import TestHumanoidEnv
from rllab.envs.normalized_env import normalize
import matplotlib.pyplot as plt
import numpy as np

# auto save config

# simple version
#experiment_spec = "100X50X25_simplehumanoid_TRPO_GAE"
# test version
experiment_spec = "128X64X32_testhumanoid_TRPO_GAE"

save_policy_every = 432

# show result config
iter_each_policy = 100
max_path_len = 5000

# test env
env = normalize(TestHumanoidEnv())
# print("action space: ")
# print(env.action_space)
# print("observation space: ")
# print(env.observation_space)

# simple env
#env = normalize(SimpleHumanoidEnv())

exper_num = 0

rewards = []
all_rewards = []

while True:
    try:
        itr_str = str((exper_num+1)*save_policy_every)
        policy = pickle.load(open("model/"+experiment_spec+itr_str+".pickle","rb"))
        exper_num+=1
        tol_reward = 0
        for i in range(iter_each_policy):
            observation = env.reset()
            # print("origin observation")
            print(observation.shape)
            env.render()
            sum_reward = 0
            for t in range(max_path_len):
                # show policy
                action, _ = policy.get_action(observation)
               
                # show random
                #action = env.action_space.sample()
                #print(action)
                observation, reward, done, _ = env.step(action)
                print(observation)
                if done:
                    break
                env.render()
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
