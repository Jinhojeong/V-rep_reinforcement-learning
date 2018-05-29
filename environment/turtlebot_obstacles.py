from __future__ import print_function
import numpy as np
from env_modules import vrep
from env_modules.core import Core
import os

scene_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'scenes')


class Turtlebot_obstacles(Core):

    def __init__(self,config):
        Core.__init__(
            self,
            config,
            os.path.join(scene_dir,'turtlebot_obstacles.ttt'))
        self.d=0.115
        self.r=0.035
    
    def launch(self):
        self.vrep_launch()
        vrep.simxSynchronousTrigger(self.clientID)
        self.joint_handles=[ \
            vrep.simxGetObjectHandle( \
                self.clientID,name,vrep.simx_opmode_blocking)[1] \
                for name in ['wheel_right_joint','wheel_left_joint']]
        self.body_handle=vrep.simxGetObjectHandle(self.clientID, \
            'Turtlebot2',vrep.simx_opmode_blocking)[1]
    
    def reward(self):
        return None
    
    def step(self,action):
        vel_right=2.0*(action[0]+self.d*action[1])/self.r
        vel_left=2.0*(action[0]-self.d*action[1])/self.r
        t=vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            vrep.simxSetJointTargetVelocity(self.clientID, \
                self.joint_handles[0],vel_right,vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(self.clientID, \
                self.joint_handles[1],vel_left,vrep.simx_opmode_streaming)
            pos=vrep.simxGetObjectPosition(self.clientID, \
                self.body_handle,self.body_handle,vrep.simx_opmode_oneshot)[1]
            ori=vrep.simxGetObjectOrientation(self.clientID, \
                self.body_handle,self.body_handle,vrep.simx_opmode_oneshot)[1]
            lrf_bin=vrep.simxGetStringSignal(self.clientID, \
                'hokuyo_data',vrep.simx_opmode_streaming)[1]
            lrf=np.reshape(vrep.simxUnpackFloats(lrf_bin),[-1,3])[:,2]
        print(pos+ori)
        state=lrf
        return state
        # return state
        reward=self.reward()
        if self.state!=None:
            self.buffer.add([self.state,action,reward,state])
        self.state=state