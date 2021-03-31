import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

"""
jobs_data = [  # task = (machine_id, processing_time).
        [(0, 3), (1, 2), (2, 2)],  # Job0
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(1, 4), (2, 3)]  # Job2
    ]
"""


ini = np.matrix([[3,2,2],[2,4,1],[0,4,3]])

class RXEnv(gym.Env):
"""
	Description:
		A job shop environment, production scheduling. 	
		No real time obs.

	Observation:
		- tasks, machines
		ex: 
				mach0 mach1 mach2
			task0	3     2     2
			task1	2     4     1
			task2	0     4     3
				

	Action:
		- action is the number of transactions in this case 8 cause we 
	have 8 moves to make. task0 3 moves, task1 3 moves, task2 2 moves.

	Reward:
		- if picked the same action twice reward = -1
		- for very move reward = 1
		- when episode done reward = [FORMULA] 

"""	

	def __init__(self):
		
		self.obs_space = spaces.Box(low = 0, high = np.inf, shape=(3,3), dtype=np.float32) 
		self.act_space = spaces.Discrete(9)
	
		self.queue = Null # check if done by saving all valid steps	
		self.state = Null
		self.ps_result = {'m0':['0,0,0'], 'm1':['0,0,0'], 'm2':['0,0,0']} # progression state result ['task_id, start_time, end_time']

	def step(self, action):
		done = False
		"""
		Approach:
					
	
		Rules:
			- action value cant be the same as previous ones.		
			- done when all tasks finished
			- can only be chosen transactions that respect sequence production
		"""

		# action conversion to coordinates
		# coord saved in (cd_row, cd_col) variables
		x = 0
		for cd_row in range(len(self.state)):
			for cd_col in range(len(slef.state[cd_row])):
				if x == action: break		
				x += 1
			if x == action: break

		
		
		seq_error = False	
		# verify if sequence is being respected
		for i in range(cd_col):
			if self.state[cd_row][i] > 0.0001:
				seq_error = True	
			

		if self.state[cd_row][cd_col] <= 0.0001 or seq_error:
			reward = 0		
	
		else:
			# make state changes 
			# save progression

			# 1- if cd_col-1 > 0 product already in process check if finished
			# 2- if target processing machine > 0
			# 3- if 1,2: max(target processing machine, machine where it was )
			# 
			# lambda function to get specific values from self.ps_result

			get_ise = lambda x,y: int(x.split(',')[y]) # split task_id, started_time, ended_time
			gt_d = lambda x,y: self.ps_result['m'+str(x)][y] # get spicific data from ps_result	
			ps_len = lambda x: len(ps_result['m'+str(x)])

			# get machine id where product was processed before
			already_proc = False
			for mach_id in range(len(self.state[cd_row]-2), -1, -1):
				if self.state[cd_row][mach_id] == 0.0001: 

					# get specific task data from machine
					for mach_queue in range(len(self.ps_state['m'+str(mach_id)]-1), -1, -1):
						if get_ise(self.ps_state['m'+str(mach_id)][mach_queue], 0) == cd_row: break
					
					already_proc = True
					break
			

		
			stt_at = get_ise(gt_d(cd_col, ps_len(cd_col)-1), 2) if not already_proc else max(get_ise(gt_d(cd_col, ps_len(cd_col)-1), 2), get_ise(gt_d(mach_id, mach_queue), 2))
			end_at = stt_at + self.state[cd_row][cd_col]	
			
			self.ps_result['m'+str(cd_col)].append('{},{},{}'.format(cd_row, stt_at, end_at))


			self.state[cd_row][cd_col] = 0.0001
			self.queue += 1

			if self.queue == 8: 
				done = True
				reward = ? # considerate time spent by all jobs/tasks less time = more reward 	
			
			else: reward = 1			

		return self.state, reward, done
			

	def reset(self):
		self._destroy()
		self.queue = Null 
		self.ps_result = Null
		self.state = ini 
		
	
	def render(self):
		pass

	def close(self):
		pass
