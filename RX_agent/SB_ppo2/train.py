import gym

from time import time, ctime

from timeit import default_timer as timer
from datetime import timedelta

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# multiprocess environment
env = gym.make('RX_env:RX-v2')

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="ppo2_log/")

stt = timer()
model.learn(total_timesteps=50000, tb_log_name="graph_3x3") 
#model.learn(total_timesteps=1000000, tb_log_name="second_x_a2c", reset_num_timesteps=False) 
end = timer()

model.save("ppo2_3x3")

print("Time spent training: ", timedelta(seconds=end-stt))

