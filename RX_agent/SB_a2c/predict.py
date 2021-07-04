import tensorflow as tf
import gym 
import numpy as np

from time import time, ctime

from timeit import default_timer as timer
from datetime import timedelta

from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C 

env = gym.make('RX_env:RX-v2')
model = A2C.load("a2c_6x6")


episode = 5
while episode > 0:
	episode -= 1

	obs = env.reset('data/ft06.jss')
	reward = 0 
	step = 0 

	while True:
		step += 1
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		    
		print("Step: ",step," | State: ", obs," | Action: ", action, " | Reward: ", rewards)
		reward += rewards
		if dones:
			print("Time [", ctime(time()),"] | Total Steps: ",step ,"Total Reward: ", reward)
			env.render()
			break

