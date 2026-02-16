"""Registration for RoboSpec-Velocity-Flat-Anymal-D-v0."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="RoboSpec-Velocity-Flat-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_d_flat_env_cfg:AnymalDFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents:skrl_flat_ppo_cfg.yaml",
    },
)
