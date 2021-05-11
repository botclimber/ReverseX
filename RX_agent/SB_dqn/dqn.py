import gym
import numpy as np
from time import ctime, time

from timeit import default_timer as timer
from datetime import timedelta

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('RX_env:RX-v1')

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="dqn_log/")

stt = timer()
model.learn(total_timesteps=100000, tb_log_name="first_run")
#model.learn(total_timesteps=400000, tb_log_name="second_run", reset_num_timesteps=False)
#model.learn(total_timesteps=500000, tb_log_name="third_run", reset_num_timesteps=False)
end = timer()

#model.save("dqn_x")

#del model # remove to demonstrate saving and loading
#model = DQN.load("dqn_x")

obs = env.reset(True)

for episode in range(1):
	
	reward = 0
	step = 0
	while True:
		step += 1
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		
		print(_states)		

		reward += rewards
		print("Step: ", step," | State: ", obs," | Action: ", action, " | Reward: ", rewards )
		
		if dones:
			env.render()	
			print("TIME [",ctime(time()),"] | Total Steps: ",step," Total Reward: ", reward)
			break

print("Time spent training: ", timedelta(seconds=end-stt))
