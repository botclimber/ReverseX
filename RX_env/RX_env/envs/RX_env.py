import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

"""
Description:
	A job shop environment, production scheduling.
	No real time obs.
Observation:
	- tasks (row), machines (col), time([row][col])
	ex:
			mach0 mach1 mach2
		task0	3     2     2
		task1	2     4     1
		task2	0     4     3

Action:
	- action is the number of transactions in this case 8 cause we
have 8 moves to make. task0 3 moves, task1 3 moves, task2 2 moves.
Reward:
	- if picked the same action twice reward = 0
	- for very move reward = 1
	- when episode done reward:

		r(T) = 1000*(pow(y, Topt)/pow(y, T))
		- y = 1.025
		- Topt = Optimal time for a job to end
		- T = job that take longer to end
"""


class RXEnv(gym.Env):

	def __init__(self):

		"""
		jobs_data = [  # task = (machine_id, processing_time).
			[(0, 3), (1, 2), (2, 2)],  # Job0
			[(0, 2), (2, 1), (1, 4)],  # Job1
			[(1, 4), (2, 3)]  # Job2
		    ]
		"""
		# -------------- input data ---------------
		self.ini = np.array([[3,2,2],[2,4,1],[0,4,3]], dtype=float)

		self.rows = len(self.ini)
		self.cols = len(self.ini[0])
		self.t_tasks = self.rows * self.cols

		self.done_trigger = np.count_nonzero(self.ini > 0) - 1 # num of valid moves/choices (-1 cause 0 counts too)

		# -----------------------------------------


		self.obs_space = spaces.Box(low = 0, high = np.inf, shape=(self.rows, self.cols), dtype=np.float32)
		self.act_space = spaces.Discrete(self.t_tasks)

		self.queue = 0 # count valid steps
		self.state = None
		self.ps_result = None  # progression state result m(x): ['task_id, start_time, end_time']


	def g_operation(self):
		data = {}
		for i in range(self.cols):
			data['m'+str(i)] = ['0,0,0']

		return data


	def step(self, action):
		done = False
		"""
		Approach:


		Rules:
			- action value cant be the same as previous ones.
			- done when all tasks fself.inished
			- can only be chosen transactions that respect sequence production
		"""

		# action conversion to coordinates
		# coord saved in (cd_row, cd_col) variables
		x = 0
		kill = False
		for cd_row in range(self.rows):
			for cd_col in range(self.cols):
				if x == action:
					kill = True
					break
				x += 1
			if kill: break


		# CRITICAL (&seq)
		seq_error = False
		# verify if sequence is being respected
		for i in range(cd_col):
			if self.state[cd_row][i] > 0.01:
				seq_error = True


		if self.state[cd_row][cd_col] <= 0.01 or seq_error:
			reward = 0

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


			# - CRITICAL (&seq)
			# get machine id where product was processed before
			already_proc = False
			for mach_id in range(cd_col-1, -1, -1):
				if self.state[cd_row][mach_id] == 0.01:

					# get specific task data from machine
					for mach_queue in range(len(self.ps_result['m'+str(mach_id)])-1, -1, -1):
						if get_ise(self.ps_result['m'+str(mach_id)][mach_queue], 0) == cd_row: break

					already_proc = True
					break


			stt_at = get_ise(gt_d(cd_col, ps_len(cd_col)-1), 2) if not already_proc else max(get_ise(gt_d(cd_col, ps_len(cd_col)-1), 2), get_ise(gt_d(mach_id, mach_queue), 2))
			end_at = stt_at + self.state[cd_row][cd_col]

			self.ps_result['m'+str(cd_col)].append('{},{},{}'.format(cd_row, stt_at, end_at))


			self.state[cd_row][cd_col] = 0.01

			if self.queue == self.done_trigger:
				done = True

				# pick job that takes more time to finish
				x = 0
				for i in range(self.cols):
					time = get_ise( gt_d(i, len(self.ps_result['m'+str(i)])-1), 2)
					if  time > x:
						x = time

				reward = 100*(1/pow(1.025, x))

			else:
				self.queue += 1
				reward = 1

		return self.state, reward, done


	def reset(self):
		self.queue = 0
		self.ps_result = self.g_operation()
		self.state = np.array(self.ini)
		
		return self.state
	
	def render(self):
		print("\n")
		for i in range(self.cols):
			print('Machine ', i,' :', end = " ")
			for j in range(1,len(self.ps_result['m'+str(i)])):
				print("[",self.ps_result['m'+str(i)][j],"]", end = " ")
			print("\n")

	
	def close(self):
		pass
