import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id ='RX-v1',
    entry_point ='RX_env.envs:RXEnv',
    #max_episode_steps = 200,
    #reward_threshold = 195.0,
    #nondeterministic = True,
)

