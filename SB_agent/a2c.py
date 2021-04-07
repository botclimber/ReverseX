import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Parallel environments
env = gym.make('RX_env:RX-v0' )

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=500000)
model.save("x")

#del model # remove to demonstrate saving and loading

#model = A2C.load("a2c_cartpole")

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
