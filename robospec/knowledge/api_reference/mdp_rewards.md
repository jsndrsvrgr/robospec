# Isaac Lab MDP Rewards

Extracted from `isaaclab.envs.mdp.rewards`

Total functions: 22

---

### mdp.is_alive

**Signature:** `is_alive(env: ManagerBasedRLEnv) -> torch.Tensor`

**Description:** Reward for being alive.

**Parameters:**
- `env`: ManagerBasedRLEnv

---

### mdp.is_terminated

**Signature:** `is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor`

**Description:** Penalize terminated episodes that don't correspond to episodic timeouts.

**Parameters:**
- `env`: ManagerBasedRLEnv

---

### mdp.lin_vel_z_l2

**Signature:** `lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize z-axis base linear velocity using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.ang_vel_xy_l2

**Signature:** `ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize xy-axis base angular velocity using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.flat_orientation_l2

**Signature:** `flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize non-flat base orientation using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.base_height_l2

**Signature:** `base_height_l2(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor`

**Description:** Penalize asset height from its target using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `target_height`: float
- `asset_cfg`: SceneEntityCfg
- `sensor_cfg`: SceneEntityCfg | None

---

### mdp.body_lin_acc_l2

**Signature:** `body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize the linear acceleration of bodies using L2-kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_torques_l2

**Signature:** `joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize joint torques applied on the articulation using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_vel_l1

**Signature:** `joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor`

**Description:** Penalize joint velocities on the articulation using an L1-kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_vel_l2

**Signature:** `joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize joint velocities on the articulation using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_acc_l2

**Signature:** `joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize joint accelerations on the articulation using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_deviation_l1

**Signature:** `joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize joint positions that deviate from the default one.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_pos_limits

**Signature:** `joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize joint positions if they cross the soft limits.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_vel_limits

**Signature:** `joint_vel_limits(env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize joint velocities if they cross the soft limits.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `soft_ratio`: float
- `asset_cfg`: SceneEntityCfg

---

### mdp.applied_torque_limits

**Signature:** `applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Penalize applied torques if they cross the limits.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.action_rate_l2

**Signature:** `action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor`

**Description:** Penalize the rate of change of the actions using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv

---

### mdp.action_l2

**Signature:** `action_l2(env: ManagerBasedRLEnv) -> torch.Tensor`

**Description:** Penalize the actions using L2 squared kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv

---

### mdp.undesired_contacts

**Signature:** `undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor`

**Description:** Penalize undesired contacts as the number of violations that are above a threshold.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `threshold`: float
- `sensor_cfg`: SceneEntityCfg

---

### mdp.desired_contacts

**Signature:** `desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor`

**Description:** Penalize if none of the desired contacts are present.

**Parameters:**
- `env`: Any
- `sensor_cfg`: SceneEntityCfg
- `threshold`: float

---

### mdp.contact_forces

**Signature:** `contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor`

**Description:** Penalize contact forces as the amount of violations of the net contact force.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `threshold`: float
- `sensor_cfg`: SceneEntityCfg

---

### mdp.track_lin_vel_xy_exp

**Signature:** `track_lin_vel_xy_exp(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Reward tracking of linear velocity commands (xy axes) using exponential kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `std`: float
- `command_name`: str
- `asset_cfg`: SceneEntityCfg

---

### mdp.track_ang_vel_z_exp

**Signature:** `track_ang_vel_z_exp(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Reward tracking of angular velocity commands (yaw) using exponential kernel.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `std`: float
- `command_name`: str
- `asset_cfg`: SceneEntityCfg

---
