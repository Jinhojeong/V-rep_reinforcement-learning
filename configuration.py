class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.state_dim=40
        self.action_dim=2
        self.action_bounds=[[0.5,1.0],[0.0,-1.0]] # [max,min]
        self.gamma=0.9 # discount factor
        self.layers=[256,256,128] # [hidden1,hidden2,... ]
        self.critic_learning_rate=1e-3
        self.actor_learning_rate=1e-4
        self.tau=1e-3
        self.l2_penalty=1e-5
        self.max_buffer=1e+5
        self.batch_size=200
        self.max_step=500
        self.max_episode=10000
        self.reward_param=0.0
        self.vrep_path='/opt/vrep'
        self.autolaunch=True
        self.visualization=True
        self.solver='bullet'
        self.dt=100
        self.api_port=20000
        

config=Settings()
