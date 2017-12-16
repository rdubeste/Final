import gym
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_episode(env, W, render):
    observation = env.reset()
    total_reward = 0
    for t in range(500):
        if render:
            env.render()
        val = sigmoid(np.matmul(W, observation))
        action = 0 if val < 0.5 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if np.abs(observation[0]) > 2.4 or np.abs(observation[3]) > 12:
            break
    return total_reward


env = gym.make('CartPole-v1')
solve_time = []

for i in range(50):
    bestW = None
    bestReward = float('-inf')
    for episode in range(1000):
        W = np.random.rand(4) * 2 - 1
        reward = run_episode(env, W, False)
        print(episode, ":", reward)
        if reward > bestReward:
            bestReward = reward
            bestW = W
            if reward >= 450:
                print("solved after {} episodes".format(episode))
                solve_time.append(episode)
                break
plt.plot(solve_time)
plt.xlabel("Trial")
plt.ylabel("Time to Solve")
avg = sum(solve_time) / 50
print(avg)
plt.plot([avg] * 50)
plt.show()


