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
        self.goal_set=[[0,0]]
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
        self.goal=self.goal_set
        vrep.simxSetObjectPosition(self.clientID, \
            self.goal_handle,-1,self.goal+[0],vrep.simx_opmode_oneshot)
        self.state0=None
        self.action_prev=[0.0,0.0]
        time.sleep(0.2)
    
    def reward(self):
        return 0
    
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
            pose=vrep.simxGetObjectPosition(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1]
            orientation=vrep.simxGetObjectOrientation(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1][2]
            goal_pos=vrep.simxGetObjectPosition(self.clientID, \
                self.goal_handle,self.body_handle,vrep.simx_opmode_oneshot)[1][1:3]
            lrf_bin=vrep.simxGetStringSignal(self.clientID, \
                'hokuyo_data',vrep.simx_opmode_streaming)[1]
            lrf=vrep.simxUnpackFloats(lrf_bin)
        goal_dist=linalg.norm(goal_pos)
        goal_angle=arctan2(-goal_pos[0],goal_pos[1])
        sys.stderr.write('\r| goal:% 2.1f,% 2.1f | pose:% 2.1f,% 2.1f' \
                            %(self.goal[0],self.goal[1],pose[0],pose[1]))
        state1=lrf+self.action_prev+[goal_dist,goal_angle]
        if min(lrf)<0.2:
            done=0
            print(' | Fail')
        elif goal_dist<0.1:
            done=0
            print(' | Success')
        else:
            done=1
        return state1,done
        # return state
        reward=self.reward()
        if self.state0!=None:
            self.replay.add({'state0':self.state0, \
                             'action':action, \
                             'reward':reward, \
                             'state1':state1, \
                             'done':done})
        self.state0=state1
        self.action_prev=action