class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.state_dim=40
        self.action_dim=2
        self.actions=7
        self.action_bounds=[[0.5,1.0],[0.0,-1.0]] # [max,min]
        self.gamma=0.9 # discount factor
        self.alpha=0.5
        self.epsilon=0.999
        self.layers=[200,200] # [hidden1,hidden2,... ]
        self.critic_learning_rate=1e-3
        self.actor_learning_rate=1e-4
        self.tau=1e-3
        self.l2_penalty=1e-4
        self.max_buffer=1e+5
        self.batch_size=64
        self.max_step=100
        self.max_episode=1500
        self.reward_param=0.0
        self.vrep_path='/home/jinhojeong/Downloads/V-REP_PRO_EDU_V3_5_0_Linux'
        self.autolaunch=False
        self.visualization=True
        self.solver='bullet'
        self.dt=50
        self.api_port=19997
        

config=Settings()