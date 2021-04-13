import tensorflow as tf
import gym
import numpy as np
from time import time, ctime

from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Parallel environments
#env = make_vec_env('RX_env:RX-v0', n_envs = 4)
env = gym.make('RX_env:RX-v0')

model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="a2c_log/")
model.learn(total_timesteps=100000, tb_log_name="first_run") 
model.learn(total_timesteps=400000, tb_log_name="second_run", reset_num_timesteps=False) 
model.learn(total_timesteps=500000, tb_log_name="third_run", reset_num_timesteps=False) 

model.save("a2c_x")

#del model # remove to demonstrate saving and loading

#model = A2C.load("a2c_x")

obs = env.reset()
reward = 0
step = 0
while True:
	step += 1
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	
	print("Step: ",step," | State: ", obs)
	reward += rewards
	if dones:
		print("Time [", ctime(time()),"] | Total Steps: ",step ,"Total Reward: ", reward)
		env.render()
		break
