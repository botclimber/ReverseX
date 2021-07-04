'''
author: Daniel Silva
version: v1.x
description: Environment for JSSP Reinforcement learning (RL)
'''

import os
import gym 

from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import random
import re



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


class RXEnv(gym.Env):
	JOBS = 6 
	MACHINES = 6	

	
	def __init__(self):
		
		self.observation_space = spaces.Box(low = 0, high = np.inf, shape=(self.MACHINES + (self.JOBS * 4) + (self.JOBS * self.MACHINES), ), dtype = np.int64)
		self.action_space = spaces.Discrete(self.JOBS)
		
		self.a_step = 0 # actual step
		self.reward = None # comulative reward

		self.data = None # data
		self.state = None # states
		self.ps_result = None # final result

	
	

	
	#Generate data for training
	def seed(self):
		
		machines = [x for x in range(self.MACHINES)]		
		data = {}
	
		for x in range(self.JOBS):
			data['job'+str(x)] = []
		
			while len(data['job'+str(x)]) < 1:
				random.shuffle(machines)
				for j in machines:
					time = random.randrange(1, 150)
					data['job'+str(x)].append([ j, time])
				
		return data



	
	# set initial state	
	def ini_state(self ):
		
		state = []
		
		for x in range(self.MACHINES):
			state.append(0)

		for y in range(self.JOBS):
			state.append(self.data['job'+str(y)][0][0]) # machine
			state.append(self.data['job'+str(y)][0][1]) # time
			state.append(len(self.data['job'+str(y)])) # tasks left

			sum = 0
			for i in range(len(self.data['job'+str(y)])):
				sum += self.data['job'+str(y)][i][1]
			state.append(sum) # total production time of a specific job
                
		# show tasks time in all states	
		for x in self.data:
			diff = self.MACHINES - len(self.data[x])
			
			for i in self.data[x]:
				state.append(i[1])	
					
			if diff > 0:
				for i in range(diff):
					state.append(0)

		return state




	def g_machines_interface(self):
		data = {}
		for i in range(self.MACHINES):
			data['m'+str(i)] = ['0,0,0']

		return data


	

	def step(self, action):
		self.a_step += 1
		
		done = True
		reward = 0

		machine_idx = self.MACHINES+(4*action) # get machine index where the task will
		# be processed

		time_idx = machine_idx+1 # get index of occupied machine time from that task
		nr_oprs_idx = machine_idx+2 # get index of number operations left for that job
		tot_prod_time_idx = machine_idx+3 # get index of total production time left
		
		job_machine = self.state[machine_idx] # machine value
		job_time = self.state[time_idx]	# time value
		job_nr_oprs = self.state[nr_oprs_idx] # number of operations value
		job_tot_prod_time = self.state[tot_prod_time_idx] # total prod time value

	
		task_0_idx = (self.MACHINES + (self.JOBS * 4)) + (self.JOBS * action)		


		if job_nr_oprs > 0:
			
			# change data time in state of task in job
			for x in range(task_0_idx, (task_0_idx + self.JOBS )):
				if self.state[x] > 0:
					self.state[x] = 0
					break


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
			
			# decrement from tot production time of a specific job
			self.state[tot_prod_time_idx] -= job_time
            
			
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

			
			# check absolute time
			makespan = max(self.state[:self.MACHINES])
			
			
			# update ps_result
			# - save progression. It will be the final result scheme
			# ----	
			self.ps_result['m'+str(job_machine)].append('{}, {}, {}'.format(action, (self.state[job_machine] - job_time), self.state[job_machine]))	
			# ----

			
			# make changes, in the end verify if there is no more operation to be processed in any job, if so send done = True
			for x in range(self.MACHINES+2, (self.MACHINES + (self.JOBS * 4)), 4):
				if self.state[x] != 0:
					done = False
					break	
             
	
			reward = self.a_step / (self.MACHINES*self.JOBS)
			if done: 
				reward += 1000 * (1.025 / pow(1.025, makespan)) 
				#print("DONE REWARD: ", reward)
			
			
		self.reward += reward

		return self.state, reward, done, {}




	def render(self):

		print("\n")
		print(self.data)
		print("\n")		

		for i in range(self.MACHINES):
			print('Machine ', i,' :', end = " " )
			for j in range(1, len(self.ps_result['m'+str(i)])):
				print("[", self.ps_result['m'+str(i)][j], "]", end = " ")
			print("\n")




	def reset(self, f_name = False):

		#if self.a_step > 0:
		#	print("\n\n Last State: ",self.state," | Steps: ",self.a_step," | Reward: ", self.reward, " | Makespan: ",max(self.state[:self.MACHINES]))


		if f_name != False: self.data = jss_data(f_name).convert()
		else: self.data = jss_data("data/ft06.jss").convert()
		#else: self.data = self.seed()

		self.state = np.array( self.ini_state() ,dtype = np.int64)
		self.ps_result = self.g_machines_interface()
		
		self.a_step = 0
		self.reward = 0
		
		return self.state	



	def close(self):
		pass



