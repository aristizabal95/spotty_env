from gymnasium.envs.registration import register

from spotty_env.env import SpottyEnv

__all__ = ["SpottyEnv"]

register(
    id="aristizabal95/SpottyEnv-v0",
    entry_point="spotty_env.env:SpottyEnv",
)