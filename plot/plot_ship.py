import matplotlib.pyplot as plt
from berth_env import *
import pandas as pd

# import data
ship_pool = pd.read_excel("./data/ship_result.xlsx")
complete_time_list = {}


# key id, 长度， 货量
# value 进港时间，离港时间，位置
for index,row in ship_pool.iterrows():
    # print(row)
    key = (row["id"], row["length"], row["cargo_max"])
    # value =
    complete_time_list[key] = (row["in_time"], row["out_time"], row["pos1"])


def plt_gantt(complete_time_list):
    plt.rcParams['font.sans-serif'] = ['Simsun']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
    fontdict_task = {
        # "family": "Microsoft YaHei",
        "style": "oblique",
        "color": "black",
        "size": 12
    }
    fontdict_time = {
        # "family": "Microsoft YaHei",
        "style": "oblique",
        "color": "white",
        "size": 12
    }
    color = ['cadetblue', 'bisque', 'blueviolet', 'burlywood', 'chocolate', 'chartreuse', 'darkcyan', 'darkkhaki',
             'darkorchid', 'darkturquoise', 'darkmagenta', 'hotpink', 'lemonchiffon', 'lightsalmon', 'darkturquoise', 'firebrick']

    color = ['#FFFFCC', '#FFCC00', '#CC9909', '#663300', '#FF6600', '#CC6666', '#FF6666', '#FFCC66', '#CC3300', '#FF3333',
             '#CC3333', '#33CCCC', '#00CC99', '#009966', '#006600', '#339900', '#00CCCC', '#999966', '#66FF00', '#999999',
            '#333333', '#CC33CC', '#666FFF', '#000CCC', '#0033CC', '#CC3333', '#9999FF', '#33CCFF', '#CC6633', '#CC9966',
             'cadetblue', 'bisque', 'blueviolet', 'burlywood', 'chocolate', 'chartreuse', 'darkcyan', 'darkkhaki',
             'darkorchid', 'darkturquoise', 'darkmagenta', 'hotpink', 'lemonchiffon', 'lightsalmon', 'darkturquoise', 'firebrick']

    for k, v in complete_time_list.items():
            occupy_num = int(k[1] / INTERVAL)
            print(k, v)
            for i in range(occupy_num):
                if i == 0:
                    plt.barh(y = v[2]+i, height=1, width=v[1]-v[0], left=v[0], edgecolor='none', color=color[int(k[0])], label='货量{}'.format(int(k[2])))
                    # plt.barh(y = 1, height=1, width=2, left=v[0], edgecolor='none', color=color[k[0]], label='货量{}'.format(k[2]))

                    # plt.text(v[0], v[2], "Vessel"+str(k[0]), fontdict=fontdict_task)
                else:
                    plt.barh(y = v[2]+i, height=1, width=v[1]-v[0], left=v[0], edgecolor='none', color=color[int(k[0])])
                plt.legend(loc = (0.90, 0), prop = {'size':10}, ncol=2)

                # if k[0] < 20:
                #     plt.legend(loc = (0.90, -0.15), prop = {'size':10})
                # else:
                #     plt.legend(loc = (0.95, -0.15), prop = {'size':10})

    ylabels = []  # 生成y轴标签
    for i in range(int(SHORE_LENGTH/(INTERVAL))):
        if i % 2 == 1:
            ylabels.append("")
        else:
            ylabels.append("{}m".format(i*INTERVAL))

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()

    # set size
    maxsize = 200
    m = 0.2
    N = 4
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin+0.04, right=0.8 - margin, bottom=0.1)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.yticks(range(int(SHORE_LENGTH/INTERVAL)), ylabels, rotation=45)
    plt.title("船舶甘特图")
    plt.xlabel("停泊时间 /h")
    plt.ylabel("泊位")
    plt.savefig('ship.svg', bbox_inches='tight')

    plt.show()

plt_gantt(complete_time_list)
