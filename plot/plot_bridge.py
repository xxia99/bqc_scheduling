import matplotlib.pyplot as plt
from berth_env import *
import pandas as pd

# import data
complete_time_list = {}
with open('../data/bridge_result_0.json', 'r', encoding='utf8')as fp:
    bridge_data = json.load(fp)

# print(bridge_data["17"])


def plt_gantt(bridge_data):
    plt.rcParams['font.sans-serif'] = ['Simsun']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
    fontdict_task = {
        # "family": "Microsoft YaHei",
        "style": "oblique",
        "color": "black",
        "size": 9
    }
    fontdict_time = {
        # "family": "Microsoft YaHei",
        "style": "oblique",
        "color": "white",
        "size": 12
    }
    color = ['cadetblue', 'bisque', 'blueviolet', 'burlywood', 'chocolate', 'chartreuse', 'darkcyan', 'darkkhaki',
             'darkorchid', 'darkturquoise', 'darkmagenta', 'hotpink', 'lemonchiffon', 'lightsalmon', 'darkturquoise', 'firebrick',
             ]
    color = ['#FFFFCC', '#FFCC00', '#CC9909', '#663300', '#FF6600', '#CC6666', '#FF6666', '#FFCC66', '#CC3300', '#FF3333',
             '#CC3333', '#33CCCC', '#00CC99', '#009966', '#006600', '#339900', '#00CCCC', '#999966', '#66FF00', '#999999',
             '#333333', '#CC33CC', '#666FF', '#000CC', '#0033CC', '#CC3333', '#9999FF', '#33CCFF', '#CC6633', '#CC9966',
               '#FFFFCC', '#FFCC00', '#CC99090', '#663300', '#FF6600', '#CC6666', '#FF6666', '#FFCC66', '#CC3300', '#FF3333',
               '#CC3333', '#33CCCC', '#00CC99', '#009966', '#006600', '#339900', '#00CCCC', '#999966', '#66FF00', '#999999'
            '#333333', '#CC33CC', '#666FF', '#000CC', '#0033CC', '#CC3333', '#9999FF', '#33CCFF', '#CC6633', '#CC9966',]
    for k, v in bridge_data.items():
        k = int(k)
        print(k, v)
        for i in range(BRIDGE_NUM):

            if k == 0:
                plt.barh(y = v[i]/INTERVAL, height=0.95, width=1, left=k, edgecolor='none', color=color[i], label='岸桥{}'.format(i))
                plt.legend(loc = (0.90, 0), prop = {'size':10})
                # plt.text(v[0], v[2], "Vessel"+str(k[0]), fontdict=fontdict_task)
            else:
                plt.barh(y = v[i]/INTERVAL, height=1, width=1, left=k, edgecolor='none', color=color[i])

    ylabels = []  # 生成y轴标签
    for i in range(int(SHORE_LENGTH/(INTERVAL))):
        if i % 2 == 1:
            ylabels.append("")
        else:
            ylabels.append("{}m".format(i*INTERVAL))

    plt.yticks(range(int(SHORE_LENGTH/INTERVAL)), ylabels, rotation=45)
    plt.title("岸桥甘特图", fontsize=12)
    plt.xlabel("停泊时间 /h", fontsize=12)
    plt.ylabel("泊位", fontsize=12)
    plt.savefig('bridge.svg', bbox_inches='tight')

    plt.show()

plt_gantt(bridge_data)
