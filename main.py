from __future__ import print_function
import sys
from configuration import config

from agent.ddpg import DDPG
from environment.turtlebot_obstacles import Turtlebot_obstacles

import time
from numpy import reshape

env=Turtlebot_obstacles(config)
agent=DDPG(config)

env.launch()

def train():
    for episode in range(config.max_episode):
        env.reset()
        print('Episode:',episode)
        env.start()
        state,done=env.step([0,0])
        for step in range(config.max_step):
            action=agent.policy(reshape(state,[1,config.state_dim]))
            state,done=env.step(reshape(action,[config.action_dim]))
            if env.replay.buffersize>10:
                batch=env.replay.batch()
                agent.update(batch)
            if done==0:
                break
        if step>=config.max_step-1:
            print(' | Timeout')
    
# def test(self, savedir):
#     traj=None
#     return traj
if __name__=='__main__':
    train()
# env.launch()