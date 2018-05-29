from __future__ import print_function
from agent.ddpg import DDPG
from configuration import config
from environment.turtlebot_obstacles import Turtlebot_obstacles
# from environment.turtlebot import Turtlebot
import time
import numpy as np

env=Turtlebot_obstacles(config)
agent=DDPG(config)

env.launch()

def train():
    for episode in range(config.max_episode):
        env.reset()
        print('Episode:',episode)
        env.start()
        state=env.step([0,0])
        # print(state)
        for step in range(config.max_step):
            action=agent.policy(state.reshape([1,36]))
            state=env.step(action.reshape([2]))
            # print(state)
            # [state0,action,reward,state1]=env.batch()
            # agent.update(
            #     state0,action,reward,state1)
    
# def test(self, savedir):
#     traj=None
#     return traj

train()
# env.launch()