import gym
import numpy as np

env = gym.make('RX_env:RX-v1')


for i_episode in range(1):
	observation = env.reset()
	state = np.array(env.state)
	
	t = 0
	while True:

		print(observation)
		action = env.action_space.sample()

		observation, reward, done, info = env.step(action)
		print("step:",t," action: ",action ," | Reward: ", reward)
		if done:
			print("Episode - {}".format(i_episode))
			print("Data: \n {}".format(state))
			env.render()
			env.reset()
			break
		
		else: t += 1
env.close()
