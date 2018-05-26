from __future__ import print_function
from agent.ddpg import DDPG
from configuration import config
from environment import turtlebot_obstacle_avoidance


# env=environment
network=DDPG(config)
    
def train(self, savedir):
    for episode in range(self.config.max_episode):
        self.env.start()
        print('Episode:',episode)
        state=self.env.step([0,0])
        self.env.pause()
        for step in range(self.config.max_step):
            action=self.network.policy(state)
            self.env.start()
            state=self.env.step(action)
            self.env.pause()
            [state0,state1,reward]=self.env.batch()
            self.network.update(
                state0,action,reward,state1)
    
def test(self, savedir):
    traj=None
    return traj


# env.launch()