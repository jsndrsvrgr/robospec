**Reward Design Explanation for Franka Panda Reach Environment**
===========================================================

### 1. Reward Term Breakdown

| **Reward Term** | **Description** | **Why Chosen** | **Weight** |
| --- | --- | --- | --- |
| **end_effector_position_tracking** | Penalize distance from EE to target position | Encourage EE to reach target | **-0.2** |
| **end_effector_position_tracking_fine_grained** | Penalize distance with tanh scaling for fine tuning | Refine positioning once close | **0.1** |
| **end_effector_orientation_tracking** | Penalize difference from target orientation | Ensure correct EE orientation | **-0.1** |
| **action_rate** | Penalize large action changes (L2 norm) | Encourage smooth actions | **-0.0001** |
| **joint_vel** | Penalize high joint velocities (L2 norm) | Prevent excessive movement speeds | **-0.0001** |
| **joint_acc** | Penalize high joint accelerations (L2 norm) | Further smoothness, implicit via acceleration | **-0.0001** |
| **termination_penalty** | Penalize episode termination | Discourage premature termination | **-1.0** |

### 2. How Reward Terms Work Together

* **Primary Objective**: The combination of **end_effector_position_tracking** (coarse) and **end_effector_position_tracking_fine_grained** (fine) drives the EE to the target position smoothly.
* **Orientation Alignment**: **end_effector_orientation_tracking** ensures the EE's orientation matches the target, preventing rotational errors.
* **Smoothness and Efficiency**:
	+ **action_rate**, **joint_vel**, and **joint_acc** collectively penalize jerky, fast, or overly dynamic movements, promoting smooth arm motions.
* **Safety/Episode Management**: **termination_penalty** discourages actions leading to premature episode ends, though its impact is more situational given the other terms' priorities.

### 3. Expected Robot Behavior During Training

* **Early Training**:
	+ Erratic movements due to exploration.
	+ Frequent failures to reach targets or maintain orientation.
	+ High penalties from **action_rate**, **joint_vel**, and **joint_acc**.
* **Late Training**:
	+ Smooth, directed movements towards targets.
	+ Precise control over EE position and orientation.
	+ Infrequent terminations, with **termination_penalty** rarely triggered.

### 4. Tradeoffs in Reward Design

* **Weight Balancing**: The relatively low weights of smoothness terms (**action_rate**, **joint_vel**, **joint_acc**) might not fully prevent jerky movements if the position tracking rewards dominate too strongly. Adjusting these weights could better balance smoothness with task achievement.
* **Orientation vs. Position Priority**: The lower weight of **end_effector_orientation_tracking** might lead to slightly less emphasis on perfect orientation if position is nearly achieved. Depending on the task's requirements, this weight might need adjustment.
* **Exploration vs. Exploitation**: The design favors exploitation of known good behaviors due to the specific penalty structures. Enhancing exploration might require temporary adjustments to reward weights or the incorporation of entropy regularization in the RL algorithm.