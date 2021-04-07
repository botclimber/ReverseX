import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import ACER

# multiprocess environment
env =  gym.make('RX_env:RX-v0')

model = ACER(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("acer_x")

#del model # remove to demonstrate saving and loading

#model = ACER.load("acer_x")

obs = env.reset()
reward = 0
while True:
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	
	print("State: ", obs)
	reward += rewards
	if dones:
		
		print("Total Reward: ", reward)
		env.render()
		break
