import gym 

from time import time, ctime

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# multiprocess environment
env = gym.make('RX_env:RX-v2')

model = PPO2.load("ppo2_3x3")

episode = 10 

while episode > 0:  
	episode -= 1
	
	obs = env.reset("f03_test_1.jss")
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
