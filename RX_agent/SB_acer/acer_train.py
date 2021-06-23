import gym 

from timeit import default_timer as timer
from datetime import timedelta

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import ACER

env =  gym.make('RX_env:RX-v1')

model = ACER(MlpPolicy, env, verbose=1, tensorboard_log="acer_log")

stt = timer()
model.learn(total_timesteps=500000, tb_log_name="graph_ft10_v1") 
#model.learn(total_timesteps=1000000, tb_log_name="second_x_a2c", reset_num_timesteps=False) 
end = timer()

model.save("acer_x")

print("Time spent training: ", timedelta(seconds=end-stt))
