# Isaac Lab MDP Terminations

Extracted from `isaaclab.envs.mdp.terminations`

Total functions: 10

---

### mdp.time_out

**Signature:** `time_out(env: ManagerBasedRLEnv) -> torch.Tensor`

**Description:** Terminate the episode when the episode length exceeds the maximum episode length.

**Parameters:**
- `env`: ManagerBasedRLEnv

---

### mdp.command_resample

**Signature:** `command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor`

**Description:** Terminate the episode based on the total number of times commands have been re-sampled.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `command_name`: str
- `num_resamples`: int

---

### mdp.bad_orientation

**Signature:** `bad_orientation(env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Terminate when the asset's orientation is too far from the desired orientation limits.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `limit_angle`: float
- `asset_cfg`: SceneEntityCfg

---

### mdp.root_height_below_minimum

**Signature:** `root_height_below_minimum(env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Terminate when the asset's root height is below the minimum height.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `minimum_height`: float
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_pos_out_of_limit

**Signature:** `joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Terminate when the asset's joint positions are outside of the soft joint limits.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_pos_out_of_manual_limit

**Signature:** `joint_pos_out_of_manual_limit(env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Terminate when the asset's joint positions are outside of the configured bounds.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `bounds`: tuple[float, float]
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_vel_out_of_limit

**Signature:** `joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Terminate when the asset's joint velocities are outside of the soft joint limits.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_vel_out_of_manual_limit

**Signature:** `joint_vel_out_of_manual_limit(env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Terminate when the asset's joint velocities are outside the provided limits.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `max_velocity`: float
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_effort_out_of_limit

**Signature:** `joint_effort_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Terminate when effort applied on the asset's joints are outside of the soft joint limits.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.illegal_contact

**Signature:** `illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor`

**Description:** Terminate when the contact force on the sensor exceeds the force threshold.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `threshold`: float
- `sensor_cfg`: SceneEntityCfg

---
