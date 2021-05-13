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

# Parallel environments
#env = make_vec_env('RX_env:RX-v0', n_envs = 4)
env = gym.make('RX_env:RX-v1')

model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="a2c_log/")

stt = timer()
model.learn(total_timesteps=10000, tb_log_name="first_x_a2c") 
#model.learn(total_timesteps=1000000, tb_log_name="second_x_a2c", reset_num_timesteps=False) 
end = timer()

#model.save("a2c_x")

#del model # remove to demonstrate saving and loading

#model = A2C.load("a2c_x")

obs = env.reset("f03.jss")
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

print("Time spent training: ", timedelta(seconds=end-stt))
