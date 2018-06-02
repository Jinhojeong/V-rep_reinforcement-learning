from __future__ import print_function
import os
import sys
import time
from numpy import array,reshape,linalg,arctan2,pi,expand_dims
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
        # self.goal_set=[[1,1],[1,2],[1,0],[1,-1]]
        self.goal_set=[[6,6]]
        self.reward_param=config.reward_param
        self.action_prev=[0.0,0.0]
        self.state0=None
        self.goal_dist_prev=None
    
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
        self.goal=choice(self.goal_set)
        vrep.simxSetObjectPosition(self.clientID, \
            self.goal_handle,-1,self.goal+[0],vrep.simx_opmode_oneshot)
        self.state0=None
        self.action_prev=[0.0,0.0]
        self.goal_dist_prev=None
        self.count=0
        self.reward_sum=0.0
        time.sleep(0.2)
    
    def start(self):
        self.vrep_start()
        t=vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        self.controller([0.0,0.0])
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            lrf_bin=vrep.simxGetStringSignal(self.clientID, \
                'hokuyo_data',vrep.simx_opmode_streaming)[1]
    
    def reward(self,lrf,goal_dist,action):
        return 20*(self.goal_dist_prev-goal_dist) \
               -(1/min(lrf)-1)/10.0 \
               -self.reward_param*(1+action[1]**2)
    
    def step(self,action):
        self.count+=1
        self.controller(action)
        t=vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            pose=vrep.simxGetObjectPosition(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1]
            orientation=vrep.simxGetObjectOrientation(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_oneshot)[1][2]
            goal_pos=vrep.simxGetObjectPosition(self.clientID, \
                self.goal_handle,self.body_handle,vrep.simx_opmode_oneshot)[1][1:3]
            # vel=vrep.simxGetObjectVelocity(self.clientID, \
            #     self.body_handle,vrep.simx_opmode_streaming)
            lrf_bin=vrep.simxGetStringSignal(self.clientID, \
                'hokuyo_data',vrep.simx_opmode_streaming)[1]
            lrf=array(vrep.simxUnpackFloats(lrf_bin),dtype=float)/5.578
        goal_dist=linalg.norm(goal_pos)
        goal_angle=arctan2(-goal_pos[0],goal_pos[1])
        sys.stderr.write('\rstep:%d| goal:% 2.1f,% 2.1f | pose:% 2.1f,% 2.1f' \
                            %(self.count,self.goal[0],self.goal[1],pose[0],pose[1]))
        state1=list(lrf)+[action[0]*2,action[1]]
        state1+=[goal_dist/5.578,goal_angle/pi] \
                    if goal_dist<5.578 else \
                [1,goal_angle/pi]
        if self.goal_dist_prev!=None:
            reward=self.reward(lrf,goal_dist,action)
            self.goal_dist_prev=goal_dist
            self.reward_sum+=reward
        if min(lrf)<0.0358:
            done=1
            print(' | avg.reward:% 4.2f | Fail'%(self.reward_sum/self.count))
        elif goal_dist<0.1:
            done=1
            print(' | avg.reward:% 4.2f | Success'%(self.reward_sum/self.count))
        else:
            done=0
        if self.state0!=None:
            self.replay.add({'state0':self.state0, \
                            'action0':action, \
                            'reward':reward, \
                            'state1':state1, \
                            'done':expand_dims(done,1)})
        self.state0=state1
        self.action_prev=action
        self.goal_dist_prev=goal_dist
        return state1,done
    
    def controller(self,action):
        vel_right=2.0*(action[0]+self.d*action[1])/self.r
        vel_left=2.0*(action[0]-self.d*action[1])/self.r
        vrep.simxSetJointTargetVelocity(self.clientID, \
            self.joint_handles[0],vel_right,vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(self.clientID, \
            self.joint_handles[1],vel_left,vrep.simx_opmode_streaming)