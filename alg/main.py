from berth_env0 import BerthEnv
import json
import os
from datetime import datetime
import tempfile
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import UnifiedLogger
import time


def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def train():
    trainer = ppo.PPOTrainer(env=BerthEnv, config={
        "env_config": {
            "log": True,
            "tide": False,
            "data_dir": 'data/ship_30.xls'
        },
        "framework": "torch",
        "num_workers": 8,
        "rollout_fragment_length": 100,

        # "evaluation_interval": 1,
        # "evaluation_duration": 1,
        # "evaluation_duration_unit": "episodes",
        # "evaluation_parallel_to_training": False,
        # "in_evaluation": False,
        # "evaluation_config": {
        #     # Example: overriding env_config, exploration, etc:
        #     "env_config": {
        #         "log": False,
        #         "tide": False,
        #         "data_dir": 'data/ship_30test.xls'
        #     },
        # },
    },
    logger_creator=custom_log_creator(
        custom_path=os.path.expanduser("ray_results"),
        custom_str='')
                             )

    # policy = trainer.get_policy()
    # for i in policy.model._modules.items():
    #     print(i[1]._model)


    reward = []

    for i in range(6000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print("training episode reward: ", result["episode_reward_mean"])
        reward.append(result["episode_reward_mean"])
        # if i > 5:
        #     print("evaluation episode reward: ", result["evaluation"]["episode_reward_mean"])
        # plot

        if i % 100 == 0:
            checkpoint = trainer.save()
            checkpoint_path = checkpoint
            print("checkpoint saved at", checkpoint)
    return checkpoint_path


def test(checkpoint_path):
    tester = ppo.PPOTrainer(env=BerthEnv, config={
        "env_config": {
            "log": False,
            "tide": False,
            "data_dir": 'data/ship_30test1.xls'
        },
        "framework": "torch",
        "num_gpus": 0,
        "num_workers": 1,
        "rollout_fragment_length": 100,
        "explore": False,
},
        logger_creator=custom_log_creator(
            custom_path=os.path.expanduser("ray_results"),
            custom_str='')
                            )

    tester.restore(checkpoint_path)
    env = BerthEnv({
        "log": False,
        "tide": False,
        "data_dir": 'data/ship_30.xls'
    })
    done = False
    episode_reward = 0
    obs = env.reset()
    x = time.time()
    while not done:
        action = tester.compute_action(obs)

        obs, reward, done, info = env.step(action)
        episode_reward += reward
    y = time.time()
    print("time: ", y-x)
    print("test episode reward: ", episode_reward)


if __name__ == "__main__":
    ray.init()
    # checkpoint_path = train()
    checkpoint_path = './ray_results/_2022-09-25_14-46-02p14sjk7u/checkpoint_003801'
    test(checkpoint_path)



