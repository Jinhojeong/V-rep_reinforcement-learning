from __future__ import print_function
import os
import random
import subprocess
import time
import numpy as np
from collections import deque

from vrep_api import vrep
import scenes
from buffer import Buffer

class Environment(object):
    def __init__(self,config):
        self.vrep_sh=config.vrep_path+'/vrep.sh'
        self.viz=config.visualization
        self.scene=os.path.join(scenes.path, \
            config.scene \
                if config.scene.split('.')[-1]=='ttt' else \
            config.scene+'.ttt')
        self.port=config.api_port
        self.clientID = None
        self._dt=50
        self.buffer=Buffer(config.max_buffer)
        self.batch_size=config.batch_size
        self.state0=None
        self.state1=None
    
    def reward(self):
        return None

    # def reset(self):
    #     vrep.simxStopSimulation(
    #         self.clientID,vrep.simx_opmode_oneshot_wait)
    #     if len(self._scene) > 1:
    #         self._env_index = random.randint(0, len(self._scene))
    #     else:
    #         self._env_index = 0

    #     self._client = self._run_env()
    #     if self._client != -1:
    #         print('Connected to V-REP')
    #         vrep.simxSynchronous(self._client, True)
    #         vrep.simxStartSimulation(self._client, vrep.simx_opmode_oneshot)

    #         self._robot, self._nav = self._spawn_robot()
    #         state = self._get_state()

    #         self._prev_error = state[5]
    #         self._max_error = self._prev_error
    #         self._motion_check = []

    #         return state
    #     else:
    #         subprocess.call('pkill vrep &', shell=True)
    #         print('Couldn\'t connect to V-REP!')

    # def step(self, action):
    #     if self._normalization:
    #         action = self._rescale_action(action)
    #     self._robot.set_motor_velocities(action)

    #     vrep.simxSynchronousTrigger(self._client)
    #     vrep.simxGetPingTime(self._client)

    #     next_state = self._get_state()
    #     reward, done = self._reward(next_state)
    #     if self._normalization:
    #         next_state = self._normalize_state(next_state)

    #     return reward, next_state, done

    # def stop(self):
    #     vrep.simxStopSimulation(self._client, vrep.simx_opmode_oneshot)
    #     while vrep.simxGetConnectionId(self._client) != -1:
    #         vrep.simxSynchronousTrigger(self._client)

    def launch(self):
        vrep.simxFinish(-1)
        if self.viz:
            vrep_exec=self.vrep_sh+' -q '
            t_val = 5.0
        else:
            vrep_exec=self.vrep_sh+' -h -q '
            t_val = 1.0
        synch_mode_cmd= \
            '-gREMOTEAPISERVERSERVICE_'+str(self.port)+'_FALSE_TRUE '
        subprocess.call( \
            vrep_exec+synch_mode_cmd+self.scene+' &',shell=True)
        time.sleep(t_val)
        self.clientID=vrep.simxStart(
            '127.0.0.1',self.port,True,True,5000,5)
        body_name='nakedAckermannSteeringCar'
        joint_names = ['nakedCar_steeringLeft','nakedCar_steeringRight']
        throttle_joint = ['nakedCar_motorLeft','nakedCar_motorRight']
        self.joint_handles = [vrep.simxGetObjectHandle(self.clientID,
	        name, vrep.simx_opmode_blocking)[1] for name in joint_names]
        self.throttle_handles = [vrep.simxGetObjectHandle(self.clientID,
            name, vrep.simx_opmode_blocking)[1] for name in throttle_joint]
        self.body_handle = vrep.simxGetObjectHandle(self.clientID,
            body_name, vrep.simx_opmode_blocking)
    
    def start(self):
        vrep.simxStartSimulation(
            self.clientID, vrep.simx_opmode_oneshot_wait)
    
    def step(self,action):
        vel=action[0]
        if(action[1]!=0):
            steeringAngleLeft=np.arctan(
                self.l/(-self.d+self.l/np.tan(action[1])))
            steeringAngleRight=np.arctan(
                self.l/(self.d+self.l/np.tan(action[1])))
        else:
            steeringAngleLeft=0
            steeringAngleRight=0
        t=vrep.simxGetLastCmdTime(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t<100:
            vrep.simxSetJointTargetPosition(self.clientID,
                self.joint_handles[0],
                steeringAngleLeft,
                vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetPosition(self.clientID,
                self.joint_handles[1],
                steeringAngleRight,
                vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(self.clientID,
                self.joint_handles[0],
                vel,
                vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(self.clientID,
                self.joint_handles[1],
                vel,
                vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(self.clientID,
                self.throttle_handles[0],
                vel,
                vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(self.clientID,
                self.throttle_handles[1],
                vel,
                vrep.simx_opmode_streaming)
            self.state1=vrep.simxGetArrayParameter(
                self.clientID,None,vrep.simx_opmode_streaming)
        return self.state1
        reward=self.reward()
        self.buffer.add([self.state0,action,reward,self.state1])

    
    def reset(self):
        vrep.simxStopSimulation(
            self.clientID,vrep.simx_opmode_oneshot_wait)
    
    def pause(self):
        vrep.simxPauseSimulation(
            self.clientID,vrep.simx_opmode_oneshot_wait)
    
    def close(self):
        self.reset()
        while vrep.simxGetConnectionId(self.clientID) != -1:
            vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxFinish(self.clientID)
        self.buffer.clear()


    # def _spawn_robot(self):
    #     robot = Robot(self._client, self._robot_model['robot_streams'],
    #                   self._robot_model['robot_objects'],
    #                   self._robot_model['wheel_diameter'],
    #                   self._robot_model['body_width'], self._dt,
    #                   self._robot_model['min_velocity'],
    #                   self._robot_model['max_velocity'])

    #     vrep.simxSynchronousTrigger(self._client)
    #     vrep.simxGetPingTime(self._client)

    #     for _ in range(5):
    #         robot.get_position()
    #         vrep.simxSynchronousTrigger(self._client)

    #     if self._navigation_method == 'ideal':
    #         navigation = Ideal(self._target_position[self._env_index],
    #                            self._robot_model['wheel_diameter'],
    #                            self._robot_model['body_width'], self._dt)
    #     elif self._navigation_method == 'odometry':
    #         navigation = Odometry(robot.get_position(), self._target_position,
    #                               self._robot_model['wheel_diameter'],
    #                               self._robot_model['body_width'], self._dt)
    #     elif self._navigation_method == 'gyrodometry':
    #         navigation = Gyrodometry(robot.get_position(),
    #                                  self._target_position,
    #                                  self._robot_model['wheel_diameter'],
    #                                  self._robot_model['body_width'], self._dt)
    #     else:
    #         raise ValueError('Invalid nevigation method')

    #     return robot, navigation

    # def _get_state(self):
    #     self._nav.compute_position(
    #         position=self._robot.get_position(),
    #         phi=self._robot.get_encoders_values(),
    #         angular_velocity=self._robot.get_gyroscope_values())

    #     dist = self._robot.get_proximity_values()
    #     error = self._nav.navigation_error

    #     return np.concatenate((dist, error))

if __name__ == '__main__':
    # config.visualization=True
    env = Environment(None)

    # env.launch()