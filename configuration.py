class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.state_dim=36
        self.action_dim=2
        self.action_bounds=[[0.5,1.0],[0.0,-1.0]]
        self.gamma=0.99
        self.layers=[200,200]
        self.critic_learning_rate=1e-3
        self.actor_learning_rate=1e-4
        self.tau=1e-3
        self.l2_penalty=1e-4
        self.max_buffer=1e+5
        self.batch_size=64
        self.max_step=200
        self.max_episode=1500
        self.reward_param=0.0
        self.vrep_path='/opt/vrep'
        self.visualization=False
        self.solver='bullet'
        self.dt=50
        self.scene=None
        self.api_port=20000
        

config=Settings()