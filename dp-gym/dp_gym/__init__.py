from gym.envs.registration import register

register(
    id='dp_gym-v0',
    entry_point='dp_gym.envs:dp_gym',
)
