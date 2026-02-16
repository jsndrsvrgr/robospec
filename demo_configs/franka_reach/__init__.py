"""Registration for RoboSpec-Reach-Franka-v0."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="RoboSpec-Reach-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_panda_reach_env_cfg:FrankaPandaReachEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents:skrl_ppo_cfg.yaml",
    },
)
