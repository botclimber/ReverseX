'''
author: Daniel Silva
version: v1.21
description: Environment for JSSP Reinforcement learning (RL)
'''

import os
import gym 

from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import random
import re


# upload data from .jss files
class jss_data():

	def __init__(self, f_name):
		self.jss_data = open(f_name, 'r').read()


	def clean(self):

		data = re.split('\n',self.jss_data)
		data_cp = data.copy()

		c = 0   
		for x in range(len(data)):
			if re.search('#', data_cp[x]) or not data_cp[x]:
				data.pop(x-c)   
				c += 1
		data.pop(0)    

		return data 

	def convert(self):
		c_data = self.clean()    

		data = {}
		for x in range(len(c_data)):
			p_data = c_data[x].split()
			data['job'+str(x)] = []

			for i in range(0, len(p_data), 2): 
				data['job'+str(x)].append([int(p_data[i]), int(p_data[i+1])])

		return data


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

JOBS = 6 
MACHINES = 6

class RXEnv(gym.Env):
	
	def __init__(self):
		
		self.observation_space = spaces.Box(low = 0, high = np.inf, shape=(MACHINES + (JOBS * 3) + 1, ), dtype = np.int64)
		self.action_space = spaces.Discrete(JOBS)
		
		self.a_step = None # count steps and give us the actual one
		self.reward = None # sum rewards

		self.data = None # data got from jss file
		
		self.state = None # actual state
		self.ps_result = None # final presentation result


	# seed with random data
	'''
	def seed(self):
		# Generate data for training
		
		machines = [x for x in range(MACHINES)]		
		data = {}
	
		for x in range(JOBS):
			data['job'+str(x)] = []
		
			while len(data['job'+str(x)]) < 1:
				random.shuffle(machines)
				for j in machines:
					time = random.randrange(0, 10)
					if time > 0: data['job'+str(x)].append([ j, time])
				
		return data
	'''

	# get initial state from given data
	def ini_state(self ):
		'''
		Generate initial state	
		'''
		state = []
		
		for x in range(MACHINES):
			state.append(0)

		for y in range(JOBS):
			state.append(self.data['job'+str(y)][0][0])
			state.append(self.data['job'+str(y)][0][1])
			state.append(len(self.data['job'+str(y)]))
		state.append(0)

		return state


	# generate dictionary with machines to presentate when episode done
	def g_machines_interface(self):
		data = {}
		for i in range(MACHINES):
			data['m'+str(i)] = ['0,0,0']

		return data


	

	def step(self, action):
		self.a_step += 1
		
		done = True
		reward = 0

		machine_idx = MACHINES+(3*action)	
		time_idx = machine_idx+1
		nr_oprs_idx = machine_idx+2
		
		job_machine = self.state[machine_idx]
		job_time = self.state[time_idx]	
		job_nr_oprs = self.state[nr_oprs_idx]		

		
		if job_nr_oprs > 0:

			# increment in machine
			# - check diff time between last opr machine and actual machine.
			# - make diff between start and end in machine where operation will be processed
			diff = 0
			if job_nr_oprs < len(self.data['job'+str(action)]):
				
				str_conv = lambda x, y: int(x.split(',')[y])				
				mach_before = self.data['job'+str(action)][-(job_nr_oprs+1)][0]
				
				for x in self.ps_result['m'+str(mach_before)]:
				
					if str_conv(x, 0) == action:
						end_time = str_conv(x, 2)

				if end_time > self.state[job_machine]:
					diff = end_time - self.state[job_machine] 
								
		
			self.state[job_machine] += (diff + job_time)	
			
			# decrement opr in state
			job_nr_oprs -= 1		
			self.state[nr_oprs_idx] = job_nr_oprs
				
			# change next opr in the specified job (mahcine, time)
			if job_nr_oprs > 0:
				self.state[machine_idx] = self.data['job'+str(action)][-job_nr_oprs][0]
				self.state[time_idx] = self.data['job'+str(action)][-job_nr_oprs][1]		

			else:
				self.state[machine_idx] = 0
				self.state[time_idx] = 0

			
			# update ps_result
			# - save progression. It will be the final result scheme
			# ----	
			self.ps_result['m'+str(job_machine)].append('{}, {}, {}'.format(action, (self.state[job_machine] - job_time), self.state[job_machine]))	
			# ----

			# make changes, in the end verify if there is no more operation to be processed in any job, if so send done = True
			for x in range(MACHINES+2, len(self.state), 3):
				if self.state[x] != 0:
					done = False
					break	
			
			
			reward = self.a_step / (MACHINES*JOBS)
			if done: 
				reward += 1.025 / pow(1.025, self.state[-1]) 
				#print("DONE REWARD: ", reward)
			

		self.reward += reward
		return self.state, reward, done, {}




	def render(self):

		print("\n")
		print(self.data)
		print("\n")		

		for i in range(MACHINES):
			print('Machine ', i,' :', end = " " )
			for j in range(1, len(self.ps_result['m'+str(i)])):
				print("[", self.ps_result['m'+str(i)][j], "]", end = " ")
			print("\n")




	def reset(self, f_name = False):
		
		if self.a_step > 0:	
			print("Last State: ",self.state," | Steps: ",self.a_step," | Reward: ", self.reward, " | Makespan: ",max(self.state[:MACHINES]))
		
		self.reward = 0		
		self.l_time = 0		
		
		if f_name != False: self.data = jss_data(f_name).convert() 
		else: self.data = jss_data("ft06.jss").convert()
		
		#self.data = jss_data("f03.jss").convert()

		#print(self.data," | ", end = " ")
		# self.static_data = self.data

		self.state = np.array( self.ini_state() ,dtype = np.int64)
		self.ps_result = self.g_machines_interface()		
		
		self.a_step = 0
		# self.max_invalid_steps = MAX_INVALID_STEPS		
	
		return self.state
	

	def close(self):
		pass



