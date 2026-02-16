from gymnasium.envs.registration import register

register(
    id="aristizabal95/SpottyEnv-v0",
    entry_point="spotty_env.env:SpottyEnv",
)