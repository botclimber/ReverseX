'''
author: Daniel Silva
version: v1.x
description: -

'''

import os
import gym 

from gym import error, spaces, utils
from gym.utils import seeding

import csv 
import numpy as np
import random

'''
data format:
	- dict
	- {'job(x)':[ [mach, time] ]dd}

google example:
{'job0':[[0,3],[1,2],[2,2]],'job1':[[0,2],[2,1],[1,4]],'job2':[[1,4],[2,3]]}


state (format ex: 3x3):
[
(m0, m1, m2 | tempo ate maquina ficar disponivel), 
(job0, job0, job0 | maquina operacao seguinte, temp opr seguinte, num oprs ainda por processar), 
(job1, job1, job1 | ", ", "),
(job2, job2, job2 | ", ", "),
(absTime | tempo absoluto, menor tempo livre de todas as maquinas) 
]
- state array length: [machines + (jobs * 3) + 1]

action: discrete( nr of jobs)

reward system:
	- if job with no more op to be processed r = -1
	- if no more op to be processed in any job (equaction)

 
'''

JOBS = 3
MACHINES = 3 

class RXEnv(gym.Env):
	
	def __init__(self):
		
		self.observation_space = spaces.Box(low = 0, high = np.inf, shape(MACHINES Â+ (JOBS * 3) + 1, ), dtype = np.int64)
		self.action_space = spaces.Discrete(JOBS)
	
		self.data = None
		self.static_data = None	
		
		self.state = None
		self.ps_result = None


	def seed(self):
		'''
		Generate data for training
		'''
		machines = [x for x in range(MACHINES)]		
		data = {}
	
		for x in range(JOBS):
			data['job'+str(x)] = []
			
			random.shuffle(machines)
			for j in machines:
				time = random.randrange(0, 10)
				if time > 0: data['job'+str(x)].append([ j, time])
				
		return data


	def ini_state(self, data):
		'''
		Generate initial state	
		'''
		state = []
		
		for x in range(MACHINES):
			state.append(0)

		for x in range(3):
			for y in range(JOBS):
				state.append(data['job'+str(y)][0][0])
				state.append(data['job'+str(y)][0][1])
				state.append(len(data['job'+str(y)]))
		state.append(0)

		return state



	def step(self, action):
		pass

	def render(self):
		pass		

	def reset(self):
		self.data = seed()
		self.static_data = self.data
	
		self.state = np.array( self.ini_state() ,dtype = np.int64)

	def close(self):
		pass
