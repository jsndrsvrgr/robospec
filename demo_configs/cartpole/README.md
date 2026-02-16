**Reward Design Explanation for Cartpole Environment**
=====================================================

### 1. Reward Term Breakdown

| **Term** | **Description** | **Why Chosen** | **Weight** |
| --- | --- | --- | --- |
| **1. Alive** | Constant reward for being "alive" (episode not terminated) | Encourages survival | **1.0** |
| **2. Terminating** | Penalty for episode termination | Discourages failure | **-2.0** |
| **3. Pole Pos** | Reward for keeping pole upright (target: 0 radians) | Primary task objective | **-1.0** (L2 distance penalty) |
| **4. Cart Vel** | Penalty for high cart velocity (L1) | Prevents excessive movement | **-0.01** |
| **5. Pole Vel** | Penalty for high pole angular velocity (L2) | Stabilizes pole movement | **-0.005** |
| **6. Action Penalty** | Penalty for large actions (L2) | Encourages smooth control | **-0.001** |

### 2. How Reward Terms Work Together

* **Survival & Failure**: "Alive" and "Terminating" terms create a baseline incentive to survive, with a significant penalty for failing.
* **Primary Task Enforcement**: "Pole Pos" directly incentivizes the main goal of keeping the pole upright.
* **Shaping Behaviors**:
	+ "Cart Vel" and "Pole Vel" terms subtly guide the agent towards controlled movements.
	+ "Action Penalty" smooths the control policy, preventing jerky actions.
* **Weight Hierarchy**: Weights are set to prioritize the primary task ("Pole Pos") and survival ("Alive" & "Terminating") over shaping behaviors.

### 3. Expected Robot Behavior During Training

* **Early Training**:
	+ Frequent terminations due to pole falls or cart going out of bounds.
	+ Erratic, high-velocity movements as the agent explores action space.
* **Late Training**:
	+ More consistent pole balance, with occasional minor wobbles.
	+ Smoother, controlled cart movements to adjust pole position.

### 4. Tradeoffs in Reward Design

* **Over-regularization**: Aggressive weights for "Cart Vel", "Pole Vel", or "Action Penalty" might overly restrict the agent's ability to learn effective movements.
* **Insufficient Shaping**: If these terms are too lightly weighted, the agent might focus solely on the primary task, leading to inefficient or dangerous (high-velocity) behaviors.
* **Delicate Balance**: The "Pole Pos" weight must be high enough to prioritize the main task without overwhelming the learning signal from other terms.