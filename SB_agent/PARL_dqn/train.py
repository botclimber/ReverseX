#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from parl.utils import check_version_for_fluid  # requires parl >= 1.4.1
check_version_for_fluid()

import gym
import numpy as np
import parl
import os.path
from RX_agent import RXagent
from RX_model import RXmodel
from parl.utils import logger

LEARNING_RATE = 1e-3


def run_episode(env, agent, train_or_test='train'):
	obs_list, action_list, reward_list = [], [], []
	obs = env.reset()
	while True:
		obs_list.append(obs)
		if train_or_test == 'train':
			action = agent.sample(obs)
		else:
			action = agent.predict(obs)
		action_list.append(action)

		obs, reward, done, info = env.step(action)
		reward_list.append(reward)

		if done:
			env.render()	
			break
	
	return obs_list, action_list, reward_list


def calc_reward_to_go(reward_list):
	for i in range(len(reward_list) - 2, -1, -1):
		reward_list[i] += reward_list[i + 1]
	return np.array(reward_list)


def main():
	env = gym.make("RX_env:RX-v0")
	model = RXmodel(act_dim=env.action_space.n)
	alg = parl.algorithms.PolicyGradient(model, lr=LEARNING_RATE)
	agent = RXagent(alg, act_dim=env.action_space.n, gamma=0.99, lr=5e-4)

	# if the file already exists, restore parameters from it
	if os.path.exists('./model_dir'):
		agent.restore('./model_dir')

	for i in range(1000):
		obs_list, action_list, reward_list = run_episode(env, agent)
		if i % 10 == 0:
			logger.info("Episode {}, Reward Sum {}.".format(
			i, sum(reward_list)))

		batch_obs = np.array(obs_list)
		batch_action = np.array(action_list)
		batch_reward = calc_reward_to_go(reward_list)

		agent.learn(batch_obs, batch_action, batch_reward)
		if (i + 1) % 100 == 0:
			_, _, reward_list = run_episode(env, agent, train_or_test='test')
			total_reward = np.sum(reward_list)
			logger.info('Test reward: {}'.format(total_reward))

	# save the parameters to ./model_dir
	agent.save('./model_dir')


if __name__ == '__main__':
	main()
