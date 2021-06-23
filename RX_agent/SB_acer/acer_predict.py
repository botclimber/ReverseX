import gym 

from time import time, ctime

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import ACER

env =  gym.make('RX_env:RX-v1')
model = ACER.load("acer_x")


episode = 10
while episode > 0:
	obs = env.reset("ft06.jss")

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
			episode -= 1
			break
