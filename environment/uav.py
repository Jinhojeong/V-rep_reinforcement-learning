from __future__ import print_function
import os
import sys
import time
from numpy import reshape,linalg,arctan2
from random import choice
from env_modules import vrep
from env_modules.core import Core


scene_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'scenes')


class Turtlebot_obstacles(Core):

    def __init__(self,config):
        Core.__init__(
            self,
            config,
            os.path.join(scene_dir,'turtlebot_obstacles.ttt'))
        self.d=0.115
        self.r=0.035
        self.starting_pose_set=[[1,1],[1,2],[1,0],[1,-1]]
        self.state0=None
        self.action_prev=[0.0,0.0]
    
    def launch(self):
        self.vrep_launch()
        vrep.simxSynchronousTrigger(self.clientID)
        self.joint_handles=[ \
            vrep.simxGetObjectHandle( \
                self.clientID,name,vrep.simx_opmode_blocking)[1] \
                for name in ['wheel_right_joint','wheel_left_joint']]
        self.body_handle=vrep.simxGetObjectHandle(self.clientID, \
            'Turtlebot2',vrep.simx_opmode_blocking)[1]
        self.goal_handle=vrep.simxGetObjectHandle(self.clientID, \
            'Goal',vrep.simx_opmode_blocking)[1]
    
    def reset(self):
        self.vrep_reset()
        self.goal=choice(self.starting_pose_set)
        vrep.simxSetObjectPosition(self.clientID, \
            self.body_handle,-1,self.goal+[0],vrep.simx_opmode_blocking)
        self.state0=None
        self.action_prev=[0.0,0.0]
        time.sleep(0.2)
        t=vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            self.controller([0,0])
    
    def reward(self):
        return 0
    
    def step(self,action):
        self.controller(action)
        t=vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            pose=vrep.simxGetObjectPosition(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1]
            orientation=vrep.simxGetObjectOrientation(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1][2]
            goal_pose=vrep.simxGetObjectPosition(self.clientID, \
                self.goal_handle,self.body_handle,vrep.simx_opmode_oneshot)[1][1:3]
        r=linalg.norm(goal_pose)
        goal_angle=arctan2(-goal_pose[0],goal_pose[1])
        r_dot=(r-self.state0[0])/self.dt
        sys.stderr.write('\r| r=% 2.1f,r_dot=% 2.1f' \
                            %(r,r_dot))
        state1=[r,r_dot]
        return state1,self.reward()
        # if self.state0!=None:
        #     self.replay.add({'state0':self.state0, \
        #                      'action':action, \
        #                      'reward':reward, \
        #                      'state1':state1, \
        #                      'done':done})
        # self.state0=state1
        # self.action_prev=action

    def controller(self,action):
        vel_right=2.0*(action[0]+self.d*action[1])/self.r
        vel_left=2.0*(action[0]-self.d*action[1])/self.r
        vrep.simxSetJointTargetVelocity(self.clientID, \
            self.joint_handles[0],vel_right,vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(self.clientID, \
            self.joint_handles[1],vel_left,vrep.simx_opmode_streaming)