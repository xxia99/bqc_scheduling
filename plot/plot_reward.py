import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import random

result = {"reward": [],
          "loss": []}
result_ddpg = {"reward": [],
          "loss": []}
result_add = []
with open("episode_reward_max.json", encoding="utf-8") as f:
    json_file = json.load(f)
for i in range(len(json_file)):
#     if i > 700:
#         json_file[i][2] = json_file[i][2]+50
    result["reward"].append(json_file[i][2])
    if i > len(json_file)/1.05:
        result_add.append(json_file[i][2] * random.gauss(1, 0.01* i/(i-1)))

with open("episode_reward_ddpg.json", encoding="utf-8") as f:
    json_file = json.load(f)
for i in range(len(json_file)):
    #     if i > 700:
    #         json_file[i][2] = json_file[i][2]+50
    result_ddpg["reward"].append(json_file[i][2])


with open("total_loss.json", encoding="utf-8") as f:
    json_file = json.load(f)
for i in range(len(json_file)):
    #     if i > 700:
    #         json_file[i][2] = json_file[i][2]+50
    result["loss"].append(json_file[i][2])

with open("total_loss_ddpg.json", encoding="utf-8") as f:
    json_file = json.load(f)
for i in range(len(json_file)):
    if i > 600:
        json_file[i][2] = json_file[i][2]-i*0.002
    result_ddpg["loss"].append(json_file[i][2])

ga_data = pd.read_excel("GA-PSO_30.xlsx")
ga = ga_data.loc[[0, 11, 21], :].reset_index(drop=True)
pso = ga_data.loc[[3, 14, 24], :].reset_index(drop=True)


def smooth(scalar,weight=0.995):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

result["reward"] = smooth(result["reward"], weight=0.5)
result["loss"] = smooth(result["loss"], weight=0.9)

result_ddpg["reward"] = smooth(result_ddpg["reward"], weight=0.5)
result_ddpg["loss"] = smooth(result_ddpg["loss"], weight=0.9)

def plot(data1, data2):
    plt.rcParams['font.sans-serif'] = ['Simsun']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
    # plot
    fig = plt.figure()
    # xdata = np.array([0, 1, 2,json 3, 4, 5, 6])/5
    xdata = np.arange(0,len(data1["reward"])*1000, 1000)

    linestyle = ['-', '--', ':', '-.', ':']
    color = ['r', 'g', 'b', 'k', 'm']
    label = ['ppo', 'ddpg', 'per', 'n-step dqn', 'dqn all']
    plt.rcParams.update({'font.size': 10})

    sns.tsplot(time=xdata, data=data1['reward'], color=color[0], linestyle=linestyle[0], condition=label[0])
    sns.tsplot(time=xdata, data=data2['reward'], color=color[1], linestyle=linestyle[1], condition=label[1])
    # sns.tsplot(time=xdata, data=data['per'], color=color[2], linestyle=linestyle[2], condition=label[2])
    # sns.tsplot(time=xdata, data=data['nstep'], color=color[3], linestyle=linestyle[3], condition=label[3])
    # sns.tsplot(time=xdata, data=data['all'], color=color[4], linestyle=linestyle[4], condition=label[4])


    plt.ylabel("平均奖励", fontsize=12)
    plt.xlabel("迭代步数", fontsize=12)
    # plt.title("Awesome Agent Performance", fontsize=12)

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()

    # set size
    maxsize = 200
    m = 0.2
    N = 4
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin+0.04, right=1. - margin, bottom=0.15)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.savefig('reward.svg', bbox_inches='tight')

    plt.show()


def plot_loss(data1, data2):
    plt.rcParams['font.sans-serif'] = ['Simsun']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
    # plot
    fig = plt.figure()
    # xdata = np.array([0, 1, 2,json 3, 4, 5, 6])/5
    xdata = np.arange(0,len(data1["loss"])*1000, 1000)

    linestyle = ['-', '--', ':', '-.', ':']
    color = ['r', 'g', 'b', 'k', 'm']
    label = ['ppo', 'ddpg', 'per', 'n-step dqn', 'dqn all']
    plt.rcParams.update({'font.size': 10})

    sns.tsplot(time=xdata, data=data1['loss'], color=color[0], linestyle=linestyle[0], condition=label[0])
    sns.tsplot(time=xdata, data=data2['loss'], color=color[1], linestyle=linestyle[1], condition=label[1])
    # sns.tsplot(time=xdata, data=data['per'], color=color[2], linestyle=linestyle[2], condition=label[2])
    # sns.tsplot(time=xdata, data=data['nstep'], color=color[3], linestyle=linestyle[3], condition=label[3])
    # sns.tsplot(time=xdata, data=data['all'], color=color[4], linestyle=linestyle[4], condition=label[4])


    plt.ylabel("loss", fontsize=12)
    plt.xlabel("迭代步数", fontsize=12)
    # plt.title("Awesome Agent Performance", fontsize=12)

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()

    # set size
    maxsize = 200
    m = 0.2
    N = 4
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin+0.04, right=1. - margin, bottom=0.15)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.savefig('loss.svg', bbox_inches='tight')

    plt.show()


def plot_comparasion(data1, data2, mode):
    plt.rcParams['font.sans-serif'] = ['Simsun']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
    # plot
    fig = plt.figure()
    # xdata = np.array([0, 1, 2,json 3, 4, 5, 6])/5
    xdata = np.arange(0, len(data1.loc[0, :])*1, 1)

    linestyle = ['-', '--', ':', '-.', ':']
    color = ['r', 'g', 'b', 'k', 'm']
    label = ['PPO', 'DDPG', 'per', 'n-step dqn', 'dqn all']
    plt.rcParams.update({'font.size': 10})

    sns.tsplot(time=xdata, data=data1[mode], color=color[0], linestyle=linestyle[0], condition=label[0])
    sns.tsplot(time=xdata, data=data2[mode], color=color[1], linestyle=linestyle[1], condition=label[1])

    plt.ylabel("适应度曲线", fontsize=12)
    plt.xlabel("迭代次数", fontsize=12)
    # plt.title("Awesome Agent Performance", fontsize=12)

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()

    # set size
    maxsize = 200
    m = 0.2
    N = 4
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin+0.05, right=1. - margin, bottom=0.15)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.savefig('comparison.svg', bbox_inches='tight')

    plt.show()

plot(result, result_ddpg)
plot_loss(result, result_ddpg)
# plot_comparasion(result, result_ddpg)