# Isaac Lab MDP Observations

Extracted from `isaaclab.envs.mdp.observations`

Total functions: 28

---

### mdp.base_pos_z

**Signature:** `base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Root height in the simulation world frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.base_lin_vel

**Signature:** `base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Root linear velocity in the asset's root frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.base_ang_vel

**Signature:** `base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Root angular velocity in the asset's root frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.projected_gravity

**Signature:** `projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Gravity projection on the asset's root frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.root_pos_w

**Signature:** `root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Asset root position in the environment frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.root_quat_w

**Signature:** `root_quat_w(env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Asset root orientation (w, x, y, z) in the environment frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `make_quat_unique`: bool
- `asset_cfg`: SceneEntityCfg

---

### mdp.root_lin_vel_w

**Signature:** `root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Asset root linear velocity in the environment frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.root_ang_vel_w

**Signature:** `root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** Asset root angular velocity in the environment frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.body_pose_w

**Signature:** `body_pose_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** The flattened body poses of the asset w.r.t the env.scene.origin.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.body_projected_gravity_b

**Signature:** `body_projected_gravity_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** The direction of gravity projected on to bodies of an Articulation.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_pos

**Signature:** `joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** The joint positions of the asset.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_pos_rel

**Signature:** `joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** The joint positions of the asset w.r.t. the default joint positions.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_pos_limit_normalized

**Signature:** `joint_pos_limit_normalized(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** The joint positions of the asset normalized with the asset's joint limits.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_vel

**Signature:** `joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** The joint velocities of the asset.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_vel_rel

**Signature:** `joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** The joint velocities of the asset w.r.t. the default joint velocities.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.joint_effort

**Signature:** `joint_effort(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor`

**Description:** The joint applied effort of the robot.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.height_scan

**Signature:** `height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor`

**Description:** Height scan from the given sensor w.r.t. the sensor's frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `sensor_cfg`: SceneEntityCfg
- `offset`: float

---

### mdp.body_incoming_wrench

**Signature:** `body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor`

**Description:** Incoming spatial wrench on bodies of an articulation in the simulation world frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.imu_orientation

**Signature:** `imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('imu')) -> torch.Tensor`

**Description:** Imu sensor orientation in the simulation world frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.imu_projected_gravity

**Signature:** `imu_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('imu')) -> torch.Tensor`

**Description:** Imu sensor orientation w.r.t the env.scene.origin.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.imu_ang_vel

**Signature:** `imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('imu')) -> torch.Tensor`

**Description:** Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.imu_lin_acc

**Signature:** `imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg('imu')) -> torch.Tensor`

**Description:** Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

**Parameters:**
- `env`: ManagerBasedEnv
- `asset_cfg`: SceneEntityCfg

---

### mdp.image

**Signature:** `image(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg('tiled_camera'), data_type: str = 'rgb', convert_perspective_to_orthogonal: bool = False, normalize: bool = True) -> torch.Tensor`

**Description:** Images of a specific datatype from the camera sensor.

**Parameters:**
- `env`: ManagerBasedEnv
- `sensor_cfg`: SceneEntityCfg
- `data_type`: str
- `convert_perspective_to_orthogonal`: bool
- `normalize`: bool

---

### mdp.last_action

**Signature:** `last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor`

**Description:** The last input action to the environment.

**Parameters:**
- `env`: ManagerBasedEnv
- `action_name`: str | None

---

### mdp.generated_commands

**Signature:** `generated_commands(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor`

**Description:** The generated command from command term in the command manager with the given name.

**Parameters:**
- `env`: ManagerBasedRLEnv
- `command_name`: str | None

---

### mdp.current_time_s

**Signature:** `current_time_s(env: ManagerBasedRLEnv) -> torch.Tensor`

**Description:** The current time in the episode (in seconds).

**Parameters:**
- `env`: ManagerBasedRLEnv

---

### mdp.remaining_time_s

**Signature:** `remaining_time_s(env: ManagerBasedRLEnv) -> torch.Tensor`

**Description:** The maximum time remaining in the episode (in seconds).

**Parameters:**
- `env`: ManagerBasedRLEnv

---

### mdp.reset

**Signature:** `reset(self, env_ids: torch.Tensor | None = None)`

**Description:** No description available.

**Parameters:**
- `env_ids`: torch.Tensor | None

---
