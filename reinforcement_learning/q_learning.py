from environment import MountainCar
import numpy as np
import sys

# load data
# mode = sys.argv[1]
# weight_out = sys.argv[2]
# returns_out = sys.argv[3]
# episodes = int(sys.argv[4])
# max_iterations = int(sys.argv[5])
# epsilon = float(sys.argv[6])
# gamma = float(sys.argv[7])
# learning_rate = float(sys.argv[8])

mode = "tile"
weight_out = "weight.out"
returns_out = "returns.out"
episodes = 400
max_iterations = 200
epsilon = 0.05
gamma = 0.99
learning_rate = 0.00005

# press ctrl+shift+c to show the terminal
# - python q_learning.py raw raw_weight.out raw_returns.out 20 200 0.05 0.99 0.01
# - python q_learning.py tile tile_weight.out tile_returns.out 20 200 0.05 0.99 0.00005


# write files
def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


# compute q-value
def q_value(s, a, wt, bias):
    dot_prod = 0
    column = wt[:, a]
    for k, v in s.items():
        dot_prod += v * column[k]
    dot_prod += bias
    return dot_prod


# epsilon-greedy action selection
def choose_action(state, epsilon, action_space, wt, bias):
    if np.random.uniform(0, 1) < epsilon:  # exploration
        action = np.random.choice(action_space)
    else:
        tmp_q_list = []
        for i in action_space:
            tmp_q_list.append(q_value(state, i, wt, bias))
        action = np.argmax(tmp_q_list)
    return action


# compute the gradient of weights; length equals to state spaces
def gradient_wt(s, a, car):
    grad_w = np.zeros((car.state_space, car.action_space))
    for key in s.keys():
        grad_w[key, a] = s[key]
    return grad_w


# get max q-value
def get_max_q(action_space, wt, bias, next_state):
    tmp_q_list = []
    for i in action_space:
        tmp_q_list.append(q_value(next_state, i, wt, bias))
    return max(tmp_q_list)


# Training for Q-learning
def Q_learning(car, wt, bias, action_space,
    episodes, max_iterations, epsilon, gamma, learn_rate):
    total_rewards = []
    rolling_mean = []
    for i in range(episodes):
        # get initial random states
        state = car.reset()
        reward_tmp = 0
        for j in range(max_iterations):
            # get optimal action with epsilon-greedy method
            action = choose_action(state, epsilon, action_space, wt, bias)
            # get reward and next state
            next_state, reward, done = car.step(action)
            
            reward_tmp += reward
            current_q = q_value(state, action, wt, bias)
            max_q = get_max_q(action_space, wt, bias, next_state)
            target = reward + gamma * max_q

            grad_w = gradient_wt(state, action, car)
            grad_b = 1

            update = learn_rate * (current_q - target)
            
            wt = wt - update * grad_w
            bias = bias - update * grad_b

            state = next_state

            if done:
                break

        total_rewards.append(reward_tmp)
        if (i >= 25):
            print(i)
            rolling_mean.append(sum(total_rewards[i-25:i]) / 25)
        else:  # i < 25
            rolling_mean.append(np.mean(total_rewards[: i+1]))
    return (wt, bias, total_rewards, rolling_mean)


# main
car = MountainCar(mode)
init_wt = np.zeros([car.state_space, car.action_space])  # 2*3 or 2048*3
init_bias = 0

action_space = list(range(car.action_space))

wt, bias, rewards, rolling_mean = Q_learning(car, init_wt, init_bias, action_space,
    episodes, max_iterations, epsilon, gamma, learning_rate)

wt_lst = np.reshape(wt, car.state_space * car.action_space)
wt_lst = np.hstack([bias, wt_lst])

# print(rewards)
# print(wt_lst)

# output results
rewards_str = ""
for i in range(len(rewards)):
    rewards_str += str(rewards[i]) + "\n"

wt_str = ""
for i in range(len(wt_lst)):
    wt_str += str(wt_lst[i]) + "\n"
writeFile(returns_out, rewards_str)
writeFile(weight_out, wt_str)

# rolling mean fill in the last 24 episodes

# plot episodes vs rewards
import matplotlib.pyplot as plt
episodes_num = list(range(1, episodes+1))
plt.plot(episodes_num, rewards, '-b', label='total rewards per episode')
plt.plot(episodes_num, rolling_mean, '-r', label='rolling mean over a 25 episode window')
plt.ylim(top=0)
plt.xlabel("number of episodes")
plt.ylabel("rewards")
plt.legend(loc='upper right')
plt_title = "Total rewards and rolling mean of the " + str(mode) + " features"
plt.title(plt_title)
# save image
plot_name = "graph_feature_" + str(mode) + ".png"
plt.savefig(plot_name)  # should before show method
# show
plt.show()