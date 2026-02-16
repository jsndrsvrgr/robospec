# Reward Engineering Patterns for Isaac Lab

## General Principles
- Primary objectives get positive weights (0.5 — 2.0)
- Penalties/constraints get negative weights (-0.0001 — -0.1)
- Regularization terms get very small negative weights (-0.0001 — -0.001)
- Always include at least one termination penalty

## Manipulation (Reach)
- Target reaching: use position_command_error_tanh, weight 1.0, std=0.05 for ~5cm precision
- Smooth motion: action_rate_l2, weight -0.01
- Joint speed regularization: joint_vel_l2, weight -0.0001
- Anti-jerk: joint_acc_l2, weight -0.0001

## Locomotion (Walking)
- Velocity tracking: track_lin_vel_xy_exp, weight 1.0-1.5
- Turn rate: track_ang_vel_z_exp, weight 0.5-0.75
- Anti-bounce: lin_vel_z_l2, weight -2.0
- Stability: ang_vel_xy_l2, weight -0.05
- Upright: flat_orientation_l2, weight -0.1 to -5.0
- Gait: feet_air_time, weight 0.1-0.5
- Safety: undesired_contacts, weight -1.0

## Classic Control (Cartpole)
- Survival: is_alive, weight 1.0
- Pole angle: joint_pos deviation from upright, weight -1.0 to -2.0
- Calm: joint_vel penalty, weight -0.01
- Efficiency: action_l2 penalty, weight -0.01
