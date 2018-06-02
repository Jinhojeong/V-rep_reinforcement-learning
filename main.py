from __future__ import print_function
import os,sys,time
from numpy import reshape,save
from configuration import config

from agent.ddpg import DDPG
from environment.turtlebot_obstacles import Turtlebot_obstacles


# env=Turtlebot_obstacles(config)
# agent=DDPG(config)

# env.launch()

def train(port=20000):
    env=Turtlebot_obstacles(config)

    agent=DDPG(config)

    env.launch()

    config.api_port=port
    for episode in range(config.max_episode):
        env.reset()
        print('Episode:',episode+1)
        env.start()
        state,done=env.step([0,0])
        for step in range(config.max_step):
            action=agent.policy(reshape(state,[1,config.state_dim]))
            state,done=env.step(reshape(action,[config.action_dim]))
            if env.replay.buffersize>10:
                batch=env.replay.batch()            
                # agent.update(batch)
            if done==0:
                break
        if step>=config.max_step-1:
            print(' | Timeout')
        if (episode+1)%100==0:
            save(os.path.join( \
                'savedir','weight_'+str(config.reward_param)+'.npy'), \
                agent.return_variables())
# def test(self, savedir):
#     traj=None
#     return traj
if __name__=='__main__':
    train()
# env.launch()