import os
import gym

from gym import error, spaces, utils
from gym.utils import seeding

import csv
import numpy as np
import random

"""
Description:
	A job shop environment, production scheduling.
	No real time obs.

Observation:
	- tasks (row), machines (col), time(min).order
	ex:
			mach0 mach1 mach2
		job0	3.1   2.2     2.3
		job1	2.1   4.3     1.2
		job2	0     4.1     3.2

Action:	
	- is the number of all matrix index. it's done by assumption that all 
index will have values wich represent time and order. if not index value will
be 0. 0 represents no process in that machine (col)

Reward:
	- if picked the same action twice, no process valid option or no 
sequence respect, reward = -1
	- for every valid move/choice reward = 1
	- when episode done reward:

		r(T) = 1000*(pow(y, Topt)/pow(y, T))
		
		- y = 1.025
		- Topt = Optimal time for a job to end
		- T = job that take longer to end
"""


# -------------- input data ---------------
# ex: np.array([[3.1,2.2,2.3],[2.1,4.3,1.2],[0,4.1,3.2]], dtype=float)
#
# - self.ROWS (jobs)
# - self.COLS (machines)

MAX_INVALID_STEPS = 1e3

def load_data():
	data = []
	r_size = 0
	c_size = 0
	
	with open('../data/data.csv', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',') 
		    
		for row in reader:
			r_size += 1
			for x in range(len(row)):
				row[x] = row[x].replace(' ','.')
				row[x] = float(row[x])
		    
			data.append(row)
	
	c_size = r_size
	return data, r_size, c_size

# -----------------------------------------




class RXEnv(gym.Env):

	def __init__(self):

		"""
		google ex: jobs_data = [  # task = (machine_id, processing_time).
				[(0, 3), (1, 2), (2, 2)],  # Job0
				[(0, 2), (2, 1), (1, 4)],  # Job1
				[(1, 4), (2, 3)]  # Job2
		    		]
		"""
		self._input, self.ROWS, self.COLS = load_data()		
		
		self.observation_space = spaces.Box(low = 0, high = np.inf, shape=(self.ROWS, self.COLS), dtype=np.float32)
		self.action_space = spaces.Discrete(self.ROWS*self.COLS)

		self.max_invalid_steps = 0 # MAX_INVALID_STEPS 
		self.max_valid_steps = 0 # num of valid moves/choices (-1 cause 0 counts too)	
		
		self.state = None
		self.ps_result = None  # progression state result m(x): ['task_id, start_time, end_time']
		
	

	def g_machines_interface(self):
		data = {}
		for i in range(self.COLS):
			data['m'+str(i)] = ['0,0,0']

		return data


	def step(self, action):
		done = True if not self.max_invalid_steps else False
	
		# get order from time.order
		g_order = lambda x: round(x - int(x), 3)
		
		"""

		Rules:
			- action value cant be the same as previous ones.
			- action cant be one where index value = 0
			- can only be chosen transactions that respect sequence production
			- done when all tasks chosen or MAX_INVALID_STEPS spent
		"""

		# action conversion to indexes
		# indexes saved in (cd_row, cd_col) variables
		x = 0
		kill = False
		for cd_row in range(self.ROWS):
			for cd_col in range(self.COLS):
				if x == action:
					kill = True
					break
				x += 1
			if kill: break


		seq_error = False
		# verify if sequence is being respected
		for i in range(self.COLS):
			d = g_order(self.state[cd_row][i])
			if self.state[cd_row][i] >= 1 and d < g_order(self.state[cd_row][cd_col]):
				seq_error = True

		

		if self.state[cd_row][cd_col] < 1 or seq_error: 
			self.max_invalid_steps -= 1
			reward = -1
		
		else:
			# make state changes
			# save progression

			# 1- if cd_col-1 > 0 product already in process check if fself.inished
			# 2- if target processing machine > 0
			# 3- if 1,2: max(target processing machine, machine where it was )
			#
			# lambda function to get specific values from self.ps_result

			get_ise = lambda x,y: float(x.split(',')[y]) # split task_id, started_time, ended_time
			gt_d = lambda x,y: self.ps_result['m'+str(x)][y] # get spicific data from ps_result
			ps_len = lambda x: len(self.ps_result['m'+str(x)])


			# get machine id where product was processed before
			already_proc = False
			for mach_id in range(self.COLS):
				
				d = round(g_order(self.state[cd_row][cd_col])-0.1, 3)
				if g_order(self.state[cd_row][mach_id]) == d  :
					
					# get specific task data from machine
					for mach_queue in range(len(self.ps_result['m'+str(mach_id)])-1, -1, -1):
						if get_ise(self.ps_result['m'+str(mach_id)][mach_queue], 0) == cd_row: break

					already_proc = True
					break
	
			"""
			[stt_at, end_at]
				- stt_at (started_at) 
				- end_at (ended_at)
			

			stt_at: is the longest time between the actual machine last job processed time and 
		the machine where this specific job was processed.
			end_at: is the stt_at plus time the job needs to be processed. stt_at + time 
		(->time<- .order).
			"""
			stt_at = get_ise(gt_d(cd_col, ps_len(cd_col)-1), 2) if not already_proc else max(get_ise(gt_d(cd_col, ps_len(cd_col)-1), 2), get_ise(gt_d(mach_id, mach_queue), 2))
			end_at = stt_at + int(self.state[cd_row][cd_col])

			self.ps_result['m'+str(cd_col)].append('{},{},{}'.format(cd_row, int(stt_at), int(end_at)))

			# update self.state
			self.state[cd_row][cd_col] = g_order(self.state[cd_row][cd_col])

			if not self.max_valid_steps:
				done = True

				# pick job that takes more time to finish
				x = 0
				for i in range(self.COLS):
					time = get_ise( gt_d(i, len(self.ps_result['m'+str(i)])-1), 2)
					if  time > x:
						x = time
				
				reward = 1000*(1/pow(1.025, x))

			else:
				self.max_valid_steps -= 1
				reward = 1

		return np.array(self.state), reward, done, {}


	def reset(self):
		
		self.state = np.array(self._input, dtype=float)
		self.max_valid_steps = np.count_nonzero(self.state > 0) - 1
		self.max_invalid_steps = MAX_INVALID_STEPS
		self.ps_result = self.g_machines_interface()
	
		#print(self.state," \n\n")	
		return self.state

	

	def render(self):
		print("\n")
		for i in range(self.COLS):
			print('Machine ', i,' :', end = " ")
			for j in range(1,len(self.ps_result['m'+str(i)])):
				print("[",self.ps_result['m'+str(i)][j],"]", end = " ")
			print("\n")

	
	def close(self):
		pass
