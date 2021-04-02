import gym
env = gym.make('RX_env:RX-v0')


for i_episode in range(2):
	observation = env.reset()

	for t in range(50):

		print(observation)
		action = env.act_space.sample()

		observation, reward, done = env.step(action)
		print("step:",t," action: ",action ," | Reward: ", reward)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			env.render()
			env.reset()
			break

env.close()