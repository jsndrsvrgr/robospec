"""Registration for RoboSpec-Cartpole-v0."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="RoboSpec-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_cartpole_env_cfg:CartpoleCartpoleEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.manager_based.classic.cartpole.agents:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.cartpole.agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.cartpole.agents:skrl_ppo_cfg.yaml",
    },
)
