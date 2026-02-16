# Isaac Lab MDP Actions

Extracted from `isaaclab.envs.mdp.actions`

Total action classes: 34

---

### JointActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for the base joint action term.

---

### JointPositionActionCfg

**Inherits from:** JointActionCfg

**Description:** Configuration for the joint position action term.

---

### RelativeJointPositionActionCfg

**Inherits from:** JointActionCfg

**Description:** Configuration for the relative joint position action term.

---

### JointVelocityActionCfg

**Inherits from:** JointActionCfg

**Description:** Configuration for the joint velocity action term.

---

### JointEffortActionCfg

**Inherits from:** JointActionCfg

**Description:** Configuration for the joint effort action term.

---

### JointPositionToLimitsActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for the bounded joint position action term.

---

### EMAJointPositionToLimitsActionCfg

**Inherits from:** JointPositionToLimitsActionCfg

**Description:** Configuration for the exponential moving average (EMA) joint position action term.

---

### BinaryJointActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for the base binary joint action term.

---

### BinaryJointPositionActionCfg

**Inherits from:** BinaryJointActionCfg

**Description:** Configuration for the binary joint position action term.

---

### BinaryJointVelocityActionCfg

**Inherits from:** BinaryJointActionCfg

**Description:** Configuration for the binary joint velocity action term.

---

### AbsBinaryJointPositionActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for the absolute binary joint position action term.

---

### NonHolonomicActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for the non-holonomic action term with dummy joints at the base.

---

### DifferentialInverseKinematicsActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for inverse differential kinematics action term.

---

### OperationalSpaceControllerActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for operational space controller action term.

---

### SurfaceGripperBinaryActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for the binary surface gripper action term.

---

### BinaryJointAction

**Inherits from:** ActionTerm

**Description:** Base class for binary joint actions.

---

### BinaryJointPositionAction

**Inherits from:** BinaryJointAction

**Description:** Binary joint action that sets the binary action into joint position targets.

---

### BinaryJointVelocityAction

**Inherits from:** BinaryJointAction

**Description:** Binary joint action that sets the binary action into joint velocity targets.

---

### AbsBinaryJointPositionAction

**Inherits from:** BinaryJointAction

**Description:** Absolute Binary joint action that sets the binary action into joint position targets.

---

### JointAction

**Inherits from:** ActionTerm

**Description:** Base class for joint actions.

---

### JointPositionAction

**Inherits from:** JointAction

**Description:** Joint action term that applies the processed actions to the articulation's joints as position commands.

---

### RelativeJointPositionAction

**Inherits from:** JointAction

**Description:** Joint action term that applies the processed actions to the articulation's joints as relative position commands.

---

### JointVelocityAction

**Inherits from:** JointAction

**Description:** Joint action term that applies the processed actions to the articulation's joints as velocity commands.

---

### JointEffortAction

**Inherits from:** JointAction

**Description:** Joint action term that applies the processed actions to the articulation's joints as effort commands.

---

### JointPositionToLimitsAction

**Inherits from:** ActionTerm

**Description:** Joint position action term that scales the input actions to the joint limits and applies them to the
articulation's joints.

---

### EMAJointPositionToLimitsAction

**Inherits from:** JointPositionToLimitsAction

**Description:** Joint action term that applies exponential moving average (EMA) over the processed actions as the
articulation's joints position commands.

---

### NonHolonomicAction

**Inherits from:** ActionTerm

**Description:** Non-holonomic action that maps a two dimensional action to the velocity of the robot in
the x, y and yaw directions.

---

### PinkInverseKinematicsActionCfg

**Inherits from:** ActionTermCfg

**Description:** Configuration for Pink inverse kinematics action term.

---

### PinkInverseKinematicsAction

**Inherits from:** ActionTerm

**Description:** Pink Inverse Kinematics action term.

---

### RMPFlowActionCfg

**Inherits from:** ActionTermCfg

**Description:** No description available.

---

### RMPFlowAction

**Inherits from:** ActionTerm

**Description:** RMPFlow task space action term.

---

### SurfaceGripperBinaryAction

**Inherits from:** ActionTerm

**Description:** Surface gripper binary action.

---

### DifferentialInverseKinematicsAction

**Inherits from:** ActionTerm

**Description:** Inverse Kinematics action term.

---

### OperationalSpaceControllerAction

**Inherits from:** ActionTerm

**Description:** Operational space controller action term.

---


## Helper Functions

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** No description available.

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.IO_descriptor

**Signature:** `IO_descriptor(self) -> GenericActionIODescriptor`

**Description:** No description available.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** No description available.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** No description available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** No description available.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** No description available.

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.IO_descriptor

**Signature:** `IO_descriptor(self) -> GenericActionIODescriptor`

**Description:** The IO descriptor of the action term.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** No description available.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** No description available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** No description available.

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.IO_descriptor

**Signature:** `IO_descriptor(self) -> GenericActionIODescriptor`

**Description:** The IO descriptor of the action term.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** No description available.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** No description available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.IO_descriptor

**Signature:** `IO_descriptor(self) -> GenericActionIODescriptor`

**Description:** The IO descriptor of the action term.

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** No description available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** No description available.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** No description available.

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.IO_descriptor

**Signature:** `IO_descriptor(self) -> GenericActionIODescriptor`

**Description:** The IO descriptor of the action term.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions)`

**Description:** No description available.

**Parameters:**
- `actions`: Any

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** No description available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.hand_joint_dim

**Signature:** `hand_joint_dim(self) -> int`

**Description:** Dimension for hand joint positions.

---

### mdp.position_dim

**Signature:** `position_dim(self) -> int`

**Description:** Dimension for position (x, y, z).

---

### mdp.orientation_dim

**Signature:** `orientation_dim(self) -> int`

**Description:** Dimension for orientation (w, x, y, z).

---

### mdp.pose_dim

**Signature:** `pose_dim(self) -> int`

**Description:** Total pose dimension (position + orientation).

---

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** Dimension of the action space (based on number of tasks and pose dimension).

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** Get the raw actions tensor.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** Get the processed actions tensor.

---

### mdp.IO_descriptor

**Signature:** `IO_descriptor(self) -> GenericActionIODescriptor`

**Description:** The IO descriptor of the action term.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor) -> None`

**Description:** Process the input actions and set targets for each task.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.apply_actions

**Signature:** `apply_actions(self) -> None`

**Description:** Apply the computed joint positions based on the inverse kinematics solution.

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** Reset the action term for specified environments.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** No description available.

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.jacobian_w

**Signature:** `jacobian_w(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.jacobian_b

**Signature:** `jacobian_b(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** No description available.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** No description available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** No description available.

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** No description available.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** Apply the processed actions to the surface gripper.

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** No description available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** No description available.

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.jacobian_w

**Signature:** `jacobian_w(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.jacobian_b

**Signature:** `jacobian_b(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.IO_descriptor

**Signature:** `IO_descriptor(self) -> GenericActionIODescriptor`

**Description:** The IO descriptor of the action term.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** No description available.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** No description available.

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** No description available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---

### mdp.action_dim

**Signature:** `action_dim(self) -> int`

**Description:** Dimension of the action space of operational space control.

---

### mdp.raw_actions

**Signature:** `raw_actions(self) -> torch.Tensor`

**Description:** Raw actions for operational space control.

---

### mdp.processed_actions

**Signature:** `processed_actions(self) -> torch.Tensor`

**Description:** Processed actions for operational space control.

---

### mdp.jacobian_w

**Signature:** `jacobian_w(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.jacobian_b

**Signature:** `jacobian_b(self) -> torch.Tensor`

**Description:** No description available.

---

### mdp.IO_descriptor

**Signature:** `IO_descriptor(self) -> GenericActionIODescriptor`

**Description:** The IO descriptor of the action term.

---

### mdp.process_actions

**Signature:** `process_actions(self, actions: torch.Tensor)`

**Description:** Pre-processes the raw actions and sets them as commands for for operational space control.

**Parameters:**
- `actions`: torch.Tensor

---

### mdp.apply_actions

**Signature:** `apply_actions(self)`

**Description:** Computes the joint efforts for operational space control and applies them to the articulation.

---

### mdp.reset

**Signature:** `reset(self, env_ids: Sequence[int] | None = None) -> None`

**Description:** Resets the raw actions and the sensors if available.

**Parameters:**
- `env_ids`: Sequence[int] | None

---
