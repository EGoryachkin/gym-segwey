from gym.envs.registration import register

register(
    id='segwey-v0',
    entry_point='gym_segwey.envs:SegweyEnv',
)
register(
    id='segwey-extrahard-v0',
    entry_point='gym_segwey.envs:SegweyExtraHardEnv',
)
