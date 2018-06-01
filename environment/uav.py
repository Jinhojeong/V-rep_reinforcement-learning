from __future__ import print_function
import os
import sys
import time
from numpy import reshape,linalg,arctan2,exp
from random import choice
from env_modules import vrep
from env_modules.core import Core


scene_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'scenes')


class UAV(Core):

    def __init__(self,config):
        Core.__init__(
            self,
            config,
            os.path.join(scene_dir,'uav.ttt'))
        self.d=0.115
        self.r=0.035
        self.starting_pose_set=[[1,1],[1,2],[1,0],[1,-1]]
        self.state0=None
        self.action_prev=3
    
    def launch(self):
        self.vrep_launch()
        self.joint_handles=[ \
            vrep.simxGetObjectHandle( \
                self.clientID,name,vrep.simx_opmode_blocking)[1] \
                for name in ['wheel_right_joint','wheel_left_joint']]
        self.body_handle=vrep.simxGetObjectHandle(self.clientID, \
            'Turtlebot2',vrep.simx_opmode_blocking)[1]
        # self.goal_handle=vrep.simxGetObjectHandle(self.clientID, \
        #     'Goal',vrep.simx_opmode_blocking)[1]
    
    def reset(self):
        self.vrep_reset()
        time.sleep(0.1)
        self.pos=choice(self.starting_pose_set)
        # vrep.simxSetObjectPosition(self.clientID, \
        #     self.body_handle,-1,self.pos+[0],vrep.simx_opmode_oneshot)
        self.state0=None
        self.action_prev=3
        time.sleep(0.2)
    
    def start(self):
        self.vrep_start()
        vrep.simxSynchronousTrigger(self.clientID)
        self.controller(3)
        t=vrep.simxGetLastCmdTime(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            self.controller(3)
            pose=vrep.simxGetObjectPosition(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1]
        self.r=linalg.norm(pose)
        # t=vrep.simxGetLastCmdTime(self.clientID)
        # while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
        #     pose=vrep.simxGetObjectPosition(self.clientID, \
        #         self.body_handle,-1,vrep.simx_opmode_oneshot)[1]
    
    def reward(self, r, r_dot):
        return exp(-linalg.norm([r-0.3,r_dot])**2)
    
    def step(self,action):
        t=vrep.simxGetLastCmdTime(self.clientID)
        print(t)
        vrep.simxSynchronousTrigger(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            self.controller(action)            
            pose=vrep.simxGetObjectPosition(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1][1:3]
            orientation=vrep.simxGetObjectOrientation(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1][2]
            vel=vrep.simxGetObjectVelocity(self.clientID, \
                self.body_handle,vrep.simx_opmode_oneshot)[1][1:3]
        print(vrep.simxGetLastCmdTime(self.clientID))
        r=linalg.norm(pose)
        r_dot=(pose[0]*vel[0]+pose[1]*vel[1])/r
        sys.stderr.write('\r| r=% 2.1f,r_dot=% 2.1f\n'%(r,r_dot))
        state1=round(r*10)*11+round(r_dot*10-5)
        if r>5:
            done=1
            print(' | Fail')
        else:
            done=0
        return state1,self.reward(r,r_dot),done
        # if self.state0!=None:
        #     self.replay.add({'state0':self.state0, \
        #                      'action':action, \
        #                      'reward':reward, \
        #                      'state1':state1, \
        #                      'done':done})
        self.r=r
        # self.action_prev=action

    def controller(self,action):
        ang=(action-3)/3.0
        vel_right=2.0*(1.5+self.d*ang)/self.r
        vel_left=2.0*(1.5-self.d*ang)/self.r
        vrep.simxSetJointTargetVelocity(self.clientID, \
            self.joint_handles[0],vel_right,vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(self.clientID, \
            self.joint_handles[1],vel_left,vrep.simx_opmode_streaming)