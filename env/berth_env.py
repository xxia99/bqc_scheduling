import numpy as np
import gym
from gym import spaces
from gym.wrappers.flatten_observation import FlattenObservation
import logging
import pandas as pd
import warnings
import json

SHIP_NUM = 4
BRIDGE_NUM = 20
INTERVAL = 25
SHORE_LENGTH = 1000

logging.basicConfig(level=logging.DEBUG
                    ,filename="log.txt"
                    ,filemode="w"
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                    ,datefmt="%Y-%m-%d %H:%M:%S"
                    )


class ShoreBridge:
    """
    ### Description

    This is template for shore bridge

    """
    def __init__(self, position_init):
        self.working_efficiency = 1
        self.position_init = position_init
        self.position_curr = self.position_init
        self.move_capacity = INTERVAL
        self.range = INTERVAL

    def move(self, direction):
        if direction == 0:
            self.position_curr -= self.move_capacity
        elif direction == 1:
            self.position_curr += self.move_capacity
        else:
            pass

        return self.position_curr


class BerthEnv(gym.Env):
    """
    ### Description

    This environment corresponds to the berth allocation problem

    ### Workflow

    1.

    ### Observation Space
    | Num      | Observation           | Min  | Max            |
    |----------|-----------------------|------|----------------|
    | 0        | ship0 status          | 0   | 2              |
    | 1        | ship0 length          | 0   | length of port |
    | 2        | ship0 cargo           | 0   | Inf            |
    | 3        | ship0 preferred       | 0   | number of      |
    |          | parking location      |     | positions in shore |
    | 4-(4n-1) | repeat above for (n-1) times |
    |          |  n is the number of ships,   |
    |          |  usually is 8                |
    |4n-(4n+20)| status of the position | 0 spare | 1 occupy |
    |(4n+20)   | status of shore bridge | 0 left  | 1 right |

    ship_status:
    0 means that this position is empty, we can add new data here.
    1 means that this ship needs to be deployed.
    2 means that all cargo has been uploaded, and this ship can move away.

    why we deign state of shore as (20, 3)?
    the first column which is state[0:20, 0] represents the cargo needs to be uploaded in each position
    the second column which is state[0:20, 1] represents the capability of shore bridge in each position

    ### Action Space

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |


    ### Rewards

    ship waiting time + deviation from preferred position +

    ### Starting State

    We initialize the properties of ships based on xls file, which is simulated according to data in port

    ### Episode Termination

    When time run out or all ships has

    ### Arguments

    """
    def __init__(self, config):
        self.max_length = SHORE_LENGTH
        self.position_interval = INTERVAL
        self.num_ship = SHIP_NUM  # 4
        self.bridge_num = BRIDGE_NUM  # 10
        self.interval_num = int(self.max_length / self.position_interval)  # 20
        self.action_dimension = self.bridge_num + self.num_ship  # 10+4=14

        self.time_now = 0
        self.day = 0
        self.deploy_index = 0
        self.bridge_position_list = {}
        self.data_dir = config["data_dir"]

        self.action_low = np.zeros((self.action_dimension,), dtype=int)
        self.ship_high = (self.interval_num + 5) * np.ones((self.num_ship,), dtype=int)
        self.bridge_high = 3 * np.ones((self.bridge_num,), dtype=int)
        self.action_high = np.concatenate((self.ship_high, self.bridge_high), axis=0)
        self.action_space = spaces.Box(low=self.action_low,
                                       # low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                       high=self.action_high, dtype=np.float32)
        # self.action_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #                                high=np.array([25, 25, 25, 25, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), dtype=np.float32)
        self.state_space = spaces.Dict(
            {
                "shore": spaces.Box(low=0, high=100, shape=(1, 3 * self.interval_num), dtype=int)
            })
        for ship in range(self.num_ship):
            self.state_space["ship" + str(ship) + "_status"] = spaces.Discrete(2)
            self.state_space["ship" + str(ship) + "_property"] = spaces.Box(low=np.array([0, 0, 0]),
                                                                            high=np.array([200, 2000, 2000]),
                                                                            dtype=int)
        # self.state_space = spaces.Dict(
        #     {
        #         "ship0_status": spaces.Discrete(2),
        #         "ship0_property": spaces.Box(low=np.array([0, 0, 0]), high=np.array([200, 2000, 2000]), dtype=int),
        #         "ship1_status": spaces.Discrete(2),
        #         "ship1_property": spaces.Box(low=np.array([0, 0, 0]), high=np.array([200, 2000, 2000]), dtype=int),
        #         "ship2_status": spaces.Discrete(2),
        #         "ship2_property": spaces.Box(low=np.array([0, 0, 0]), high=np.array([200, 2000, 2000]), dtype=int),
        #         "ship3_status": spaces.Discrete(2),
        #         "ship3_property": spaces.Box(low=np.array([0, 0, 0]), high=np.array([200, 2000, 2000]), dtype=int),
        #         "shore": spaces.Box(low=0, high=100, shape=(1, 60), dtype=int)
        # })

        self.observation_space = spaces.Box(low=0, high=2000, shape=(3*self.interval_num+4*self.num_ship, ), dtype=int)

        self.log = config["log"]
        self.tide = config["tide"]

    def step(self, action_all):
        action = list(map(int, action_all[0:self.num_ship].tolist()))
        print(action)
        # initialize some parameters
        if self.log:
            logging.info("time!!! {}".format(self.day*24+self.time_now))
            logging.info("action!!! {}".format(list(map(int, action_all.tolist()))))
        reward = 0
        done = False

        # obtain ships needed to be deployed from ship_pool
        # ignore warnings of function append in dataframe
        # when tide rises,
        if self.time_now > 6 and self.time_now < 16:
            self.tide = True
        else:
            self.tide = False

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.remaining_deploy = self.remaining_deploy.append(self.ship_pool[(
                    self.ship_pool['arrival_time'] == (self.time_now+self.day*24))], ignore_index=True)
            # if self.tide:
            #     self.meet_length_ship = self.ship_pool[(self.ship_pool['arrival_time'] <= (self.time_now+self.day*24))]
            #     self.remaining_deploy = self.remaining_deploy.append(self.meet_length_ship[self.meet_length_ship['length']>100], ignore_index=True)
            # else:
            #     self.remaining_deploy = self.remaining_deploy.append(self.ship_pool[(
            #             self.ship_pool['arrival_time'] <= (self.time_now+self.day*24))], ignore_index=True)

        if self.log:
            logging.info("remaining_deploy!!! {}".format(self.remaining_deploy))
        # ship_status 0 means that this position is empty
        # ship_status 1 means that ship remains to be deployed, once this ship has been deployed, status changes to 0
        # and the ship condition shows in the shore

        for i in range(self.num_ship):
            if self.state["ship"+str(i)+"_status"] == 0 and not self.remaining_deploy.empty:
                self.state["ship"+str(i)+"_property"] = np.array([self.remaining_deploy.iloc[0]["length"],
                                                self.remaining_deploy.iloc[0]["cargo_max"],
                                                self.remaining_deploy.iloc[0]["prefer_position"]])
                self.state["ship"+str(i)+"_id"] = self.remaining_deploy.iloc[0]["id"]
                self.remaining_deploy = self.remaining_deploy.drop(labels=0).reset_index(drop=True)
                self.state["ship"+str(i)+"_status"] = 1

        # deploy the ship into shore
        for i in range(self.num_ship):
            if self.state["ship"+str(i)+"_status"] == 1 and action[i] <= self.interval_num:
                occupy_num = int(np.ceil(self.state["ship"+str(i)+"_property"][0] / INTERVAL))
                occupy_flag = self.state["shore"][action[i]:(action[i]+occupy_num), 0].any()
                if not occupy_flag and (action[i]+occupy_num) < SHORE_LENGTH/INTERVAL:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        self.ship_discharging = self.ship_discharging.append(pd.Series({
                            "id": self.state["ship"+str(i)+"_id"],
                            "cargo_curr": self.state["ship"+str(i)+"_property"][1],
                            "discharge_time": 0,
                            "position_curr": [action[i], (action[i]+occupy_num)]
                        }), ignore_index=True)
                    self.state["ship"+str(i)+"_status"] = 0
                    self.state["shore"][action[i]:(action[i]+occupy_num), 0] = np.ones((occupy_num, ))
                    index = self.ship_pool[(self.ship_pool['id'] == self.state["ship"+str(i)+"_id"])].index.values
                    self.ship_pool.loc[index[0], 'in_time'] = self.time_now + self.day*24
                    if i == 0:
                        self.num += 1
                else:
                    # if index out of range, give agent a punishment
                    reward -= 1
            else:
                pass

        # compute the shoreBridge
        action_2 = list(map(int, action_all[self.num_ship:self.action_dimension].tolist()))  # 4:14
        bridge_position = []
        threshold = INTERVAL * (self.interval_num - 1)  # 950
        for i, bridge in enumerate(self.shorebridges):
            previous_position = bridge.position_curr
            # print("index: ", i)
            # print("init: ", bridge.position_init)
            if i == 0:
                position_max = min(threshold, self.shorebridges[i+1].position_curr)
                position_min = 0
            elif i == self.bridge_num-1:  # 9
                position_max = threshold
                position_min = max(0, self.shorebridges[i-1].position_curr)
            else:
                position_max = min(threshold, self.shorebridges[i+1].position_curr)
                position_min = max(0, self.shorebridges[i-1].position_curr)

            if position_max - position_min <= 2*INTERVAL:
                bridge.move(action_2[i])
            else:
                bridge.move(action_2[i])
            # print(position_min, position_max, bridge.position_curr)

            if position_min == position_max:
                print("!!!!!")
            if bridge.position_curr <= position_min:
                bridge.position_curr = position_min + INTERVAL
            if bridge.position_curr >= position_max:
                bridge.position_curr = position_max - INTERVAL
            bridge_position.append(bridge.position_curr)
            # Give a penalty when
            if bridge.position_curr != previous_position:
                reward -= 0.1
            # print(position_min, position_max, bridge.position_curr)

        # put the information of bridges in state
        self.bridge_position_list[self.day*24+self.time_now] = list(map(float, bridge_position))

        self.state["shore"][:, 2] = np.zeros((self.interval_num, ))
        for i, bridge in enumerate(bridge_position):
            self.state["shore"][int(bridge/INTERVAL), 2] += 1

        # discharge cargo
        self.ship_discharging = self.ship_discharging.reset_index(drop=True)
        if self.ship_discharging.size == 0:
            num = 0
        else:
            num = len(self.ship_discharging.index.values)

        for i in range(num):
            # print(i)
            # print(self.ship_discharging)
            pos1 = self.ship_discharging.loc[i]["position_curr"][0]
            pos2 = self.ship_discharging.loc[i]["position_curr"][1]
            ship_position = np.arange(pos1, pos2)
            discharge_capability = np.sum(list(map(bridge_position.count, INTERVAL * ship_position)))

            self.ship_discharging.loc[i, "cargo_curr"] -= discharge_capability * 50
            # print(self.ship_discharging.loc[i, "discharge_time"])
            self.ship_discharging.loc[i, "discharge_time"] += 1

            if self.ship_discharging.loc[i]["cargo_curr"] <= 0:
                self.state["shore"][pos1: pos2, 0] = np.zeros((pos2-pos1, ))
                self.state["shore"][pos1: pos2, 1] = np.zeros((pos2-pos1, ))
                # collect the time that ships finish discharging cargo
                index = self.ship_pool[(self.ship_pool['id'] == self.ship_discharging.loc[i, "id"])].index.values
                self.ship_pool.loc[index[0], 'out_time'] = self.time_now + self.day*24 + 1
                self.ship_pool.loc[index[0], 'pos1'] = pos1
                reward += 5
                if self.ship_pool.loc[index[0], 'out_time'] - self.ship_pool.loc[index[0], 'in_time'] < self.ship_pool.loc[index[0], 'expected_time']:
                    pass
                else:
                    reward -= 3

                self.ship_discharging = self.ship_discharging.drop(labels=i)
                # print("succeed!!!")

            else:
                try:
                    self.state["shore"][pos1: pos2, 0] = np.ones((pos2-pos1, ))
                except ValueError as e:
                    raise ValueError("error!!!")
                self.state["shore"][pos1: pos2, 1] = self.ship_discharging.loc[i]["cargo_curr"] * np.ones((pos2-pos1, ))
                reward += 0.1

        # calculate the cargo
        # the third column represents shore
        # cargo_remains = self.state["shore"][:, 1] - self.state["shore"][:, 2] * 100
        # self.state["shore"][:, 1] = cargo_remains
        self.state["shore"][:, 1] = np.clip(self.state["shore"][:, 1], 0, 10000)

        if self.log:
            logging.info("state!!! {}".format(self.state))

        # episode termination
        if self.time_now == 23:
            self.time_now = 0
            self.day += 1
        else:
            self.time_now += 1
        out_time_list = self.ship_pool['out_time'].values
        if out_time_list.all():
            done = True
            self.ship_pool.to_excel('data/ship_result.xlsx', index=False, header=True)
            with open("data/bridge_result_0.json", 'w', encoding="utf-8") as f:
                json.dump(self.bridge_position_list, f)

        if self.day == 4:
            done = True
            # print(self.ship_pool)
            self.ship_pool.to_excel('data/ship_result.xlsx', index=False, header=True)
            with open("data/bridge_result_0.json", 'w', encoding="utf-8") as f:
                json.dump(self.bridge_position_list, f)

        # change the size for state to match the format of observation space
        self.obs = []
        for i in range(self.num_ship):
            self.obs.append(self.state["ship" + str(i) + "_status"])
            self.obs.extend(self.state["ship" + str(i) + "_property"].tolist())

        self.obs.extend(self.state["shore"].reshape((3*self.interval_num, )).tolist())  # 60

        reward -= 1
        return self.obs, reward, done, {}

    def reset(self):
        self.day = 0
        self.time_now = 0
        self.num = 0
        # put all ships in ship_pool
        self.ship_pool = pd.read_excel(self.data_dir)
        self.ship_pool.insert(self.ship_pool.shape[1], 'in_time', 0)
        self.ship_pool.insert(self.ship_pool.shape[1], 'out_time', 0)
        self.ship_pool.insert(self.ship_pool.shape[1], 'pos1', 0)
        self.ship_pool.insert(self.ship_pool.shape[1], 'expected_time', self.ship_pool.loc[:, "cargo_max"].values/ self.ship_pool.loc[:, "length"].values )

        # ships which are needed to be deployed in time 0
        if self.log:
            logging.info("ship_pool!!! {}".format(self.ship_pool))
        self.remaining_deploy = pd.DataFrame(columns=self.ship_pool.columns)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore")
        #     self.remaining_deploy = self.remaining_deploy.append(self.ship_pool[(self.ship_pool['arrival_time'] == self.time_now)], ignore_index=True)

        self.shorebridges = []
        for i in range(self.bridge_num):
            self.shorebridges.append(ShoreBridge(INTERVAL*(2*i+1)))

        bridge_position = []
        for i, bridge in enumerate(self.shorebridges):
            bridge_position.append(bridge.position_curr)

        self.ship_discharging = pd.DataFrame(columns=["id", "cargo_curr", "discharge_time", "position_curr"])
        self.state = {"shore": np.concatenate((np.zeros((self.interval_num, 2)),
                                               np.zeros((self.interval_num, 1))), axis=1)}  # 20
        for ship in range(self.num_ship):
            self.state["ship" + str(ship) + "_status"] = 0
            self.state["ship" + str(ship) + "_property"] = np.array(
                [2 * INTERVAL, 1000, self.interval_num])  # [100,1000,20]

        for i, bridge in enumerate(bridge_position):
            self.state["shore"][int(bridge/INTERVAL), 2] += 1

        self.obs = []
        for i in range(self.num_ship):
            self.obs.append(self.state["ship"+str(i)+"_status"])
            self.obs.extend(self.state["ship"+str(i)+"_property"].tolist())

        self.obs.extend(self.state["shore"].reshape((3*self.interval_num, )).tolist())  # 60
        return self.obs


if __name__ == '__main__':
    # create env
    env = BerthEnv({
        "log": False,
        "tide": False,
        "data_dir": 'data/ship_30.xls'
    })
    env = FlattenObservation(env)
    # env = MultiDiscreteEnv(env)
    obs = env.reset()
    for i in range(800):
        obs, reward, done, _ = env.step(np.array([5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))