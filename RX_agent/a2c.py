import gym
import numpy as np
from time import time, ctime

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Parallel environments
env = gym.make('RX_env:RX-v0')

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=500000)
model.save("a2c_x")

#del model # remove to demonstrate saving and loading

#model = A2C.load("a2c_x")

obs = env.reset()
reward = 0
step = 0
while True:
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	
	print("State: ", obs)
	step += 1
	reward += rewards
	if dones:
		print("Time [", ctime(time()),"] | Total Steps: ",step ,"Total Reward: ", reward)
		env.render()
		break
