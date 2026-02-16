# Isaac Lab MDP Events

Extracted from `isaaclab.envs.mdp.events`

Total functions: 16

---

### mdp.randomize_rigid_body_scale

**Signature:** `randomize_rigid_body_scale(env: ManagerBasedEnv, env_ids: torch.Tensor | None, scale_range: tuple[float, float] | dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg, relative_child_path: str | None = None)`

**Description:** Randomize the scale of a rigid body asset in the USD stage.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor | None
- `scale_range`: tuple[float, float] | dict[str, tuple[float, float]]
- `asset_cfg`: SceneEntityCfg
- `relative_child_path`: str | None

---

### mdp.randomize_rigid_body_com

**Signature:** `randomize_rigid_body_com(env: ManagerBasedEnv, env_ids: torch.Tensor | None, com_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg)`

**Description:** Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor | None
- `com_range`: dict[str, tuple[float, float]]
- `asset_cfg`: SceneEntityCfg

---

### mdp.randomize_rigid_body_collider_offsets

**Signature:** `randomize_rigid_body_collider_offsets(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, rest_offset_distribution_params: tuple[float, float] | None = None, contact_offset_distribution_params: tuple[float, float] | None = None, distribution: Literal['uniform', 'log_uniform', 'gaussian'] = 'uniform')`

**Description:** Randomize the collider parameters of rigid bodies in an asset by adding, scaling, or setting random values.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor | None
- `asset_cfg`: SceneEntityCfg
- `rest_offset_distribution_params`: tuple[float, float] | None
- `contact_offset_distribution_params`: tuple[float, float] | None
- `distribution`: Literal['uniform', 'log_uniform', 'gaussian']

---

### mdp.randomize_physics_scene_gravity

**Signature:** `randomize_physics_scene_gravity(env: ManagerBasedEnv, env_ids: torch.Tensor | None, gravity_distribution_params: tuple[list[float], list[float]], operation: Literal['add', 'scale', 'abs'], distribution: Literal['uniform', 'log_uniform', 'gaussian'] = 'uniform')`

**Description:** Randomize gravity by adding, scaling, or setting random values.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor | None
- `gravity_distribution_params`: tuple[list[float], list[float]]
- `operation`: Literal['add', 'scale', 'abs']
- `distribution`: Literal['uniform', 'log_uniform', 'gaussian']

---

### mdp.apply_external_force_torque

**Signature:** `apply_external_force_torque(env: ManagerBasedEnv, env_ids: torch.Tensor, force_range: tuple[float, float], torque_range: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** Randomize the external forces and torques applied to the bodies.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `force_range`: tuple[float, float]
- `torque_range`: tuple[float, float]
- `asset_cfg`: SceneEntityCfg

---

### mdp.push_by_setting_velocity

**Signature:** `push_by_setting_velocity(env: ManagerBasedEnv, env_ids: torch.Tensor, velocity_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** Push the asset by setting the root velocity to a random value within the given ranges.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `velocity_range`: dict[str, tuple[float, float]]
- `asset_cfg`: SceneEntityCfg

---

### mdp.reset_root_state_uniform

**Signature:** `reset_root_state_uniform(env: ManagerBasedEnv, env_ids: torch.Tensor, pose_range: dict[str, tuple[float, float]], velocity_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** Reset the asset root state to a random position and velocity uniformly within the given ranges.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `pose_range`: dict[str, tuple[float, float]]
- `velocity_range`: dict[str, tuple[float, float]]
- `asset_cfg`: SceneEntityCfg

---

### mdp.reset_root_state_with_random_orientation

**Signature:** `reset_root_state_with_random_orientation(env: ManagerBasedEnv, env_ids: torch.Tensor, pose_range: dict[str, tuple[float, float]], velocity_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** Reset the asset root position and velocities sampled randomly within the given ranges
and the asset root orientation sampled randomly from the SO(3).

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `pose_range`: dict[str, tuple[float, float]]
- `velocity_range`: dict[str, tuple[float, float]]
- `asset_cfg`: SceneEntityCfg

---

### mdp.reset_root_state_from_terrain

**Signature:** `reset_root_state_from_terrain(env: ManagerBasedEnv, env_ids: torch.Tensor, pose_range: dict[str, tuple[float, float]], velocity_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** Reset the asset root state by sampling a random valid pose from the terrain.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `pose_range`: dict[str, tuple[float, float]]
- `velocity_range`: dict[str, tuple[float, float]]
- `asset_cfg`: SceneEntityCfg

---

### mdp.reset_joints_by_scale

**Signature:** `reset_joints_by_scale(env: ManagerBasedEnv, env_ids: torch.Tensor, position_range: tuple[float, float], velocity_range: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** Reset the robot joints by scaling the default position and velocity by the given ranges.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `position_range`: tuple[float, float]
- `velocity_range`: tuple[float, float]
- `asset_cfg`: SceneEntityCfg

---

### mdp.reset_joints_by_offset

**Signature:** `reset_joints_by_offset(env: ManagerBasedEnv, env_ids: torch.Tensor, position_range: tuple[float, float], velocity_range: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** Reset the robot joints with offsets around the default position and velocity by the given ranges.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `position_range`: tuple[float, float]
- `velocity_range`: tuple[float, float]
- `asset_cfg`: SceneEntityCfg

---

### mdp.reset_nodal_state_uniform

**Signature:** `reset_nodal_state_uniform(env: ManagerBasedEnv, env_ids: torch.Tensor, position_range: dict[str, tuple[float, float]], velocity_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'))`

**Description:** Reset the asset nodal state to a random position and velocity uniformly within the given ranges.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `position_range`: dict[str, tuple[float, float]]
- `velocity_range`: dict[str, tuple[float, float]]
- `asset_cfg`: SceneEntityCfg

---

### mdp.reset_scene_to_default

**Signature:** `reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False)`

**Description:** Reset the scene to the default state specified in the scene configuration.

**Parameters:**
- `env`: ManagerBasedEnv
- `env_ids`: torch.Tensor
- `reset_joint_targets`: bool

---

### mdp.randomize

**Signature:** `randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor`

**Description:** No description available.

**Parameters:**
- `data`: torch.Tensor
- `params`: tuple[float, float]

---

### mdp.rep_texture_randomization

**Signature:** `rep_texture_randomization()`

**Description:** No description available.

---

### mdp.rep_color_randomization

**Signature:** `rep_color_randomization()`

**Description:** No description available.

---
