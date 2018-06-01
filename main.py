from __future__ import print_function
import sys
from configuration import config

from agent.qlearn import QLearn
from environment.uav import UAV

import time
from numpy import reshape

env=UAV(config)
agent=QLearn(config)

env.launch()

def train():
    for episode in range(config.max_episode):
        env.reset()
        print('Episode:',episode)
        env.start()
        state0,reward,done=env.step(3)
        while done==0:
            action=agent.chooseAction(state0)
            state1,reward,done=env.step(action)
            agent.learn(state0,action,reward,state1)
            state0=state1
            # if done==1:
            #     break
        # if step==config.max_step-1:
        #     print(' | Timeout')
    
# def test(self, savedir):
#     traj=None
#     return traj
if __name__=='__main__':
    train()
# env.launch()