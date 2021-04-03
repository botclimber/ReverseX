import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('RX_env:RX-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save("RX")

#del model # remove to demonstrate saving and loading
#model = DQN.load("RX")

obs = env.reset()

for episode in range(1):
	
	reward = 0
	step = 0
	while True:
		step += 1
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		
		print(_states)		

		reward += rewards
		print("Episode: ", episode," Action: ", action, " | Reward: ", rewards )
		
		if dones:
			env.render()	
			print("Total Steps: ",step," Total Reward: ", reward)
			break
