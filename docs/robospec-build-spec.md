# RoboSpec — Build Specification (48-Hour Sprint)

## Context for Claude Code

You are building **RoboSpec**, a CLI tool that takes a natural language description of a robot learning task and generates a complete, runnable NVIDIA Isaac Lab reinforcement learning environment configuration. The generated configs are real Python files that train to convergence in Isaac Lab on a cloud GPU.

**This is a contest submission for NVIDIA GTC 2026 Golden Ticket (deadline: Feb 15, 2026).**

### NVIDIA Technologies Used
1. **Nemotron 3 Nano 30B-A3B** — LLM that generates all configs (1M token context window)
2. **Isaac Lab** — The RL training framework the generated configs target
3. **NVIDIA NIM** — API endpoint for Nemotron inference

### What We're Building
- A Python CLI tool: `robospec generate "Train a Franka arm to reach targets smoothly"`
- It outputs a complete folder of Isaac Lab config files that can be run directly
- Optionally, a simple Streamlit web UI for the demo video

### What We're NOT Building (out of scope)
- No deployed web service / cloud hosting of the app
- No live training streaming
- No React frontend
- No FastAPI backend
- No refine/edit commands
- Training is done manually on AWS — not automated from the CLI

---

## 1. Project Structure

Create this exact file structure:

```
robospec/
├── README.md
├── LICENSE                          # Apache 2.0
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── robospec/
│   ├── __init__.py                  # version = "0.1.0"
│   ├── cli.py                       # Typer CLI entry point
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── analyzer.py              # NL → TaskSpec dataclass
│   │   ├── context.py               # Select few-shot examples + API docs
│   │   ├── generator.py             # Nemotron → parsed config files
│   │   ├── validator.py             # AST + semantic validation
│   │   └── explainer.py             # Generate README explaining rewards
│   │
│   ├── nemotron/
│   │   ├── __init__.py
│   │   └── client.py                # Async HTTP client for NIM + OpenRouter
│   │
│   ├── prompts/
│   │   ├── system.txt               # Base system prompt for Nemotron
│   │   ├── analyze.txt              # Task analysis prompt template
│   │   ├── generate.txt             # Config generation prompt template
│   │   └── explain.txt              # Reward explanation prompt template
│   │
│   ├── knowledge/                   # Pre-extracted Isaac Lab reference material
│   │   ├── examples/                # 5 real Isaac Lab env configs (Python files)
│   │   │   ├── cartpole_env_cfg.py
│   │   │   ├── franka_reach_env_cfg.py
│   │   │   ├── franka_reach_joint_pos_env_cfg.py
│   │   │   ├── anymal_d_flat_env_cfg.py
│   │   │   └── anymal_d_rough_env_cfg.py
│   │   │
│   │   ├── api_reference/           # Extracted function signatures + docstrings
│   │   │   ├── mdp_rewards.md
│   │   │   ├── mdp_observations.md
│   │   │   ├── mdp_actions.md
│   │   │   ├── mdp_terminations.md
│   │   │   └── mdp_events.md
│   │   │
│   │   ├── robots.json              # Robot catalog: joints, limits, assets
│   │   └── reward_patterns.md       # Common reward engineering heuristics
│   │
│   └── templates/                   # Jinja2 templates for non-LLM boilerplate
│       ├── __init__.py.j2           # Gymnasium registration template
│       └── train.sh.j2              # Training launch script template
│
├── streamlit_app.py                 # Simple demo UI (optional, for video recording)
│
└── tests/
    ├── test_analyzer.py
    ├── test_generator.py
    └── test_validator.py
```

---

## 2. Dependencies

### pyproject.toml

```toml
[project]
name = "robospec"
version = "0.1.0"
description = "Natural language to Isaac Lab RL environments"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
dependencies = [
    "typer[all]>=0.9.0",
    "httpx>=0.25.0",
    "pydantic>=2.5.0",
    "jinja2>=3.1.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
ui = ["streamlit>=1.29.0"]

[project.scripts]
robospec = "robospec.cli:app"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

### .env.example

```bash
# Get from: https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b
NVIDIA_API_KEY=nvapi-xxxxx

# Backup — get from: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-xxxxx
```

### .gitignore

```
.env
__pycache__/
*.pyc
dist/
*.egg-info/
.venv/
output/
```

---

## 3. Component Specifications

Build these components in this exact order. Each section specifies the file path, purpose, inputs, outputs, and the implementation.

---

### 3.1 Nemotron Client

**File:** `robospec/nemotron/client.py`

**Purpose:** Make API calls to Nemotron 3 Nano. Try NVIDIA NIM endpoint first, fall back to OpenRouter.

**API Details:**
- NIM endpoint: `https://integrate.api.nvidia.com/v1/chat/completions`
- NIM model string: `"nvidia/llama-3.3-nemotron-super-49b-v1"` 
- OpenRouter endpoint: `https://openrouter.ai/api/v1/chat/completions`
- OpenRouter model string: `"nvidia/llama-3.3-nemotron-super-49b-v1"`
- Both use standard OpenAI-compatible chat completions format
- Authentication: Bearer token in Authorization header

**NOTE ON MODEL SELECTION:** We are using `nvidia/llama-3.3-nemotron-super-49b-v1` as our primary model. This is NVIDIA's most capable instruction-tuned model available on NIM, excellent for code generation and structured output. If this model is unavailable, try `nvidia/nemotron-3-nano-30b-a3b` as a fallback. Check https://build.nvidia.com for current model availability before first run.

**Implementation:**

```python
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

class NemotronClient:
    NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

    def __init__(self):
        self.nim_key = os.getenv("NVIDIA_API_KEY")
        self.or_key = os.getenv("OPENROUTER_API_KEY")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> str:
        """Generate a completion. Try NIM first, fallback to OpenRouter."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.nim_key:
            try:
                return await self._call(self.NIM_URL, self.nim_key, messages, temperature, max_tokens)
            except Exception as e:
                print(f"[WARN] NIM failed: {e}, trying OpenRouter...")

        if self.or_key:
            return await self._call(self.OPENROUTER_URL, self.or_key, messages, temperature, max_tokens)

        raise RuntimeError("No API key set. Set NVIDIA_API_KEY or OPENROUTER_API_KEY in .env")

    async def _call(self, url, key, messages, temperature, max_tokens) -> str:
        resp = await self.client.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": self.MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

---

### 3.2 Task Analyzer

**File:** `robospec/pipeline/analyzer.py`

**Purpose:** Take a natural language string and produce a structured `TaskSpec`.

**How it works:** Send the user's description to Nemotron with a prompt that asks for structured JSON output. Parse the JSON into a `TaskSpec` dataclass.

**TaskSpec dataclass:**

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class TaskCategory(Enum):
    MANIPULATION_REACH = "manipulation_reach"
    CLASSIC_CARTPOLE = "classic_cartpole"
    LOCOMOTION_FLAT = "locomotion_flat"
    LOCOMOTION_ROUGH = "locomotion_rough"

class RobotType(Enum):
    FRANKA_PANDA = "franka_panda"
    CARTPOLE = "cartpole"
    ANYMAL_D = "anymal_d"

@dataclass
class TaskSpec:
    category: TaskCategory
    robot: RobotType
    description: str
    objectives: list[str]
    constraints: list[str]
    difficulty: str = "medium"
    episode_length_s: float = 5.0
    num_envs: int = 4096
    custom_notes: Optional[str] = None
```

**Prompt for analysis** (`prompts/analyze.txt`):

The prompt must instruct Nemotron to:
1. Read the user's description
2. Classify it into one of the 4 TaskCategory values
3. Select the appropriate robot
4. Extract objectives and constraints
5. Respond with ONLY a JSON object matching the TaskSpec fields

Important: The prompt must list the valid categories and robots, and must say "respond with ONLY valid JSON, no markdown fences, no explanation."

**The analyzer function:**
1. Load `prompts/analyze.txt`
2. Insert user description
3. Call `NemotronClient.generate()`
4. Parse the JSON response into a `TaskSpec`
5. If JSON parsing fails, retry once with a more explicit "respond only in JSON" instruction

---

### 3.3 Context Builder

**File:** `robospec/pipeline/context.py`

**Purpose:** Given a `TaskSpec`, assemble the relevant subset of the knowledge base into a single string to include in the generation prompt.

**Logic:**

```python
# Always include:
#   - All 5 files in api_reference/
#   - robots.json
#   - reward_patterns.md

# Conditionally include examples based on task category:
EXAMPLE_MAP = {
    "manipulation_reach": ["franka_reach_env_cfg.py", "franka_reach_joint_pos_env_cfg.py", "cartpole_env_cfg.py"],
    "classic_cartpole": ["cartpole_env_cfg.py", "franka_reach_env_cfg.py"],
    "locomotion_flat": ["anymal_d_flat_env_cfg.py", "anymal_d_rough_env_cfg.py"],
    "locomotion_rough": ["anymal_d_rough_env_cfg.py", "anymal_d_flat_env_cfg.py"],
}
```

**Output:** A single string formatted as:

```
=== ISAAC LAB API REFERENCE ===

--- mdp_rewards.md ---
[contents]

--- mdp_observations.md ---
[contents]

[... all api_reference files ...]

=== ROBOT SPECIFICATIONS ===
[robots.json contents]

=== REWARD ENGINEERING PATTERNS ===
[reward_patterns.md contents]

=== WORKING EXAMPLE CONFIGURATIONS ===
Follow these patterns exactly. These are real, working Isaac Lab configs.

--- franka_reach_env_cfg.py ---
```python
[contents]
```

[... selected examples ...]
```

---

### 3.4 Config Generator

**File:** `robospec/pipeline/generator.py`

**Purpose:** Call Nemotron with the full context + task spec and parse the multi-file output.

**Prompt template** (`prompts/generate.txt`):

The generation prompt must:
1. Set the role: "You are RoboSpec, an expert Isaac Lab environment designer"
2. List strict rules:
   - Output ONLY valid Python code
   - Follow @configclass pattern exactly as in examples
   - Only use reward/obs/termination functions from the API reference provided
   - Only use robots from the robot catalog provided
   - Include all required config classes: SceneCfg, ObservationsCfg, ActionsCfg, RewardsCfg, TerminationsCfg, EventCfg
   - Include __post_init__ with decimation, episode_length, sim.dt
3. Include the task spec fields (category, robot, objectives, constraints, difficulty, episode_length)
4. Request output in this exact format:

```
### FILE: {task_name}_env_cfg.py
[code]

### FILE: __init__.py
[code]

### FILE: train.sh
[code]
```

**GeneratedConfig dataclass:**

```python
@dataclass
class GeneratedConfig:
    env_cfg: str          # Main environment config Python code
    init_py: str          # Gymnasium registration
    train_script: str     # Shell script to launch training
    readme: str           # Explanation (generated later by explainer)
    raw_response: str     # Full Nemotron response for debugging
    task_name: str        # e.g., "franka-reach-smooth"
```

**Parsing logic:**
1. Split Nemotron's response on `### FILE:` markers
2. For each file section, extract filename and content
3. Strip any markdown code fences (```python ... ```) from content
4. If parsing fails (no ### FILE markers found), treat entire response as env_cfg

**Output:** Write all files to a directory: `output/{task_name}/`

---

### 3.5 Validator

**File:** `robospec/pipeline/validator.py`

**Purpose:** Validate generated Python code before saving.

**Validation layers (implement all):**

1. **AST Parse** — `ast.parse(code)`. If SyntaxError, the code is invalid.
2. **Class Check** — Walk AST, verify at least one class name contains "EnvCfg"
3. **Reward Check** — Walk AST, verify at least one class name contains "RewardsCfg"
4. **Post-init Check** — Verify a `__post_init__` method exists somewhere
5. **Import Check** — Verify the code imports from `isaaclab` or `omni.isaac`

**Output:**

```python
@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```

If validation fails, the generator should retry once, including the error messages in the retry prompt to Nemotron.

---

### 3.6 Explainer

**File:** `robospec/pipeline/explainer.py`

**Purpose:** Generate a human-readable explanation of the generated config's reward design.

**How:** Send the generated env_cfg code + original task description to Nemotron with a prompt asking it to explain:
- What each reward term does and why it was chosen
- How the weights were balanced
- What behavior should emerge during training

**Output:** A markdown string saved as `output/{task_name}/README.md`

---

### 3.7 CLI

**File:** `robospec/cli.py`

**Purpose:** Main entry point. User runs: `robospec generate "description"`

**Commands:**

```
robospec generate "Train a Franka arm to reach random targets smoothly"
    --output ./my_task/     (optional, default: ./output/{auto_name}/)
    --robot franka_panda    (optional, override auto-detection)
    --verbose               (optional, print full Nemotron responses)
```

**CLI flow:**

```
1. Print banner: "🤖 RoboSpec — Natural Language → Isaac Lab Environments"
2. Print: "Analyzing task..."
3. Call analyzer → get TaskSpec
4. Print: "Detected: {category} with {robot}"
5. Print: "Building context ({N} tokens of Isaac Lab reference)..."
6. Call context builder
7. Print: "Generating Isaac Lab configuration..."
8. Call generator → get GeneratedConfig
9. Print: "Validating generated code..."
10. Call validator
11. If invalid: print warnings/errors, retry once
12. Print: "Generating explanation..."
13. Call explainer → get README
14. Write all files to output directory
15. Print summary:
    "✅ Generated {N} files in {output_dir}/"
    "   - {task_name}_env_cfg.py"
    "   - __init__.py"
    "   - train.sh"
    "   - README.md"
    ""
    "To train: scp this folder to an Isaac Lab machine and run:"
    "   bash train.sh"
```

Use the `rich` library for colored output and a progress spinner during Nemotron calls.

---

## 4. Knowledge Base

### How to Populate the Knowledge Base

The knowledge base must be populated before the pipeline works. This is a manual extraction step.

**Step 1: Clone Isaac Lab**
```bash
git clone https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab
```

**Step 2: Copy example configs**

These are the few-shot examples Nemotron learns from. Copy these exact files:

```bash
# Cartpole
cp /tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py \
   robospec/knowledge/examples/

# Franka reach configs
cp /tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/joint_pos_env_cfg.py \
   robospec/knowledge/examples/franka_reach_joint_pos_env_cfg.py

cp /tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/ik_abs_env_cfg.py \
   robospec/knowledge/examples/franka_reach_env_cfg.py

# ANYmal-D locomotion
# Look in: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/
# Copy the flat terrain and rough terrain config files
```

Note: Exact file paths may vary by Isaac Lab version. Use `find /tmp/IsaacLab -name "*env_cfg*" -path "*/manager_based/*"` to locate them.

**Step 3: Extract MDP API reference**

Write a Python script that:
1. Opens each file in `/tmp/IsaacLab/source/isaaclab/isaaclab/envs/mdp/`
   - `rewards.py`
   - `observations.py`
   - `terminations.py`
   - `actions/` directory
   - `events.py`
2. For each function, extracts: function name, parameters, docstring
3. Saves as markdown files in `robospec/knowledge/api_reference/`

Format for each function:
```markdown
### mdp.{function_name}

**Signature:** `{function_name}(env, {params}) -> torch.Tensor`

**Description:** {docstring}

**Parameters:**
- {param}: {description}
```

**Step 4: Create robots.json**

```json
{
  "franka_panda": {
    "name": "Franka Emika Panda",
    "type": "manipulator",
    "dof": 7,
    "asset_path": "ISAACLAB_ASSET_DIR/Robots/FrankaPanda/franka_panda.usd",
    "joint_names": ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"],
    "ee_link": "panda_hand",
    "notes": "7-DOF robot arm. Most common for reach/grasp tasks."
  },
  "cartpole": {
    "name": "Cartpole",
    "type": "classic_control",
    "dof": 2,
    "joint_names": ["slider_to_cart", "cart_to_pole"],
    "notes": "Classic control benchmark. 1 actuated joint (cart), 1 passive (pole)."
  },
  "anymal_d": {
    "name": "ANYmal-D",
    "type": "quadruped",
    "dof": 12,
    "asset_path": "ISAACLAB_ASSET_DIR/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
    "joint_names": ["LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE", "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"],
    "notes": "Quadruped robot. 12 actuated joints (3 per leg: hip abduction, hip flexion, knee)."
  }
}
```

Note: Asset paths and joint names should be verified against the actual Isaac Lab source. The above are best-effort defaults — check `/tmp/IsaacLab/source/isaaclab_assets/` for exact values.

**Step 5: Create reward_patterns.md**

```markdown
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
```

---

## 5. Streamlit Demo UI (Optional, for video recording)

**File:** `streamlit_app.py`

**Purpose:** Simple web UI to record a polished demo video. NOT a production deployment.

**Layout:**

```
┌──────────────────────────────────────────────────────────┐
│  🤖 RoboSpec — Describe a Robot Task, Get an RL Env      │
├──────────────────────┬───────────────────────────────────┤
│                      │                                   │
│  Describe your task: │  Generated Configuration:         │
│  ┌────────────────┐  │  ┌─────────────────────────────┐ │
│  │ [text area]    │  │  │ [syntax highlighted Python] │ │
│  │                │  │  │                             │ │
│  └────────────────┘  │  │ Tabs: env_cfg | init | train│ │
│                      │  └─────────────────────────────┘ │
│  [🚀 Generate]       │                                   │
│                      │  Explanation:                     │
│  Detected:           │  ┌─────────────────────────────┐ │
│  🤖 Franka Panda     │  │ [reward explanation]        │ │
│  🎯 Reach targets    │  └─────────────────────────────┘ │
│                      │                                   │
├──────────────────────┴───────────────────────────────────┤
│  Training Demo:                                          │
│  [embedded .mp4 video of pre-recorded Isaac Lab training]│
└──────────────────────────────────────────────────────────┘
```

**Implementation:** Use `st.columns()` for the two-panel layout. Use `st.code()` with `language="python"` for syntax highlighting. Use `st.video()` for pre-recorded training videos. Use `st.tabs()` for switching between generated files.

The Streamlit app should import and call the same pipeline functions as the CLI.

---

## 6. Prompt Files — Full Content

### prompts/system.txt

```
You are RoboSpec, an AI that generates NVIDIA Isaac Lab reinforcement learning environment configurations. You are an expert in:
- Isaac Lab's Manager-Based environment workflow using @configclass decorators
- Reward engineering for robotic tasks
- The Isaac Lab mdp module (rewards, observations, terminations, events, actions)
- Robot assets available in Isaac Lab (Franka Panda, ANYmal-D, Cartpole, etc.)

You generate production-quality Python code that runs directly in Isaac Lab without modification. You only use functions and classes that exist in the Isaac Lab API — never hallucinate function names or parameters.

When generating environment configurations, you follow the exact patterns shown in the working examples provided in your context. You pay close attention to import paths, class hierarchies, and the @configclass decorator pattern.
```

### prompts/analyze.txt

```
Analyze the following robot learning task description and extract a structured specification.

VALID CATEGORIES (pick exactly one):
- manipulation_reach: Robot arm reaching or tracking target positions → Robot: franka_panda
- classic_cartpole: Cart-pole balancing or swing-up tasks → Robot: cartpole
- locomotion_flat: Quadruped walking on flat ground → Robot: anymal_d
- locomotion_rough: Quadruped walking on rough/uneven terrain → Robot: anymal_d

INSTRUCTIONS:
1. Pick the category that best matches the description
2. Extract what the robot should achieve (objectives)
3. Extract what should be penalized or avoided (constraints)
4. If the description doesn't specify constraints, infer reasonable defaults
5. Set difficulty: "easy" for simple tasks, "medium" for standard, "hard" for complex

Respond with ONLY a valid JSON object. No markdown fences. No explanation. No text before or after the JSON.

{
  "category": "manipulation_reach",
  "robot": "franka_panda",
  "objectives": ["reach random target positions"],
  "constraints": ["minimize jerk"],
  "difficulty": "medium",
  "episode_length_s": 5.0,
  "num_envs": 4096,
  "custom_notes": null
}

USER DESCRIPTION:
{user_input}
```

### prompts/generate.txt

```
Generate a complete, runnable Isaac Lab environment configuration for the following task.

STRICT RULES:
1. Output ONLY valid Python code in the specified file format below
2. Follow the @configclass pattern EXACTLY as shown in the example configurations in your context
3. ONLY use reward functions, observation functions, termination functions, and event functions that appear in the API Reference provided in your context
4. ONLY use robots that appear in the Robot Specifications in your context
5. Reward weights: positive for objectives, negative for penalties
6. You MUST include all required config classes: SceneCfg, ObservationsCfg, ActionsCfg, RewardsCfg, TerminationsCfg, EventCfg
7. You MUST include a __post_init__ method that sets decimation, episode_length_s, and sim.dt
8. The environment class MUST inherit from ManagerBasedRLEnvCfg
9. Register with gymnasium using the pattern: RoboSpec-{Robot}-{Task}-v0
10. Do NOT use any functions, classes, or imports that don't exist in Isaac Lab

TASK SPECIFICATION:
- Category: {category}
- Robot: {robot}
- Objectives: {objectives}
- Constraints: {constraints}
- Difficulty: {difficulty}
- Episode Length: {episode_length_s}s
- Num Environments: {num_envs}

Output the following files using ### FILE: markers. Each file's code must be complete and runnable.

### FILE: {task_name}_env_cfg.py
[Complete ManagerBasedRLEnvCfg with all config classes]

### FILE: __init__.py
[Gymnasium registration using gym.register()]

### FILE: train.sh
[One-line shell command to train with skrl or sb3]
```

### prompts/explain.txt

```
You are explaining the reward design of an Isaac Lab RL environment to a roboticist.

Given the following environment configuration and the original task description, explain:

1. Each reward term: what it does, why it was chosen, and what weight it has
2. How the reward terms work together to achieve the desired behavior
3. What the robot's behavior should look like during training (early vs. late)
4. Any tradeoffs in the reward design

Be concise and technical. Use a markdown format with a table for the reward terms.

TASK DESCRIPTION: {description}

GENERATED CONFIGURATION:
```python
{env_cfg}
```

Respond in markdown format.
```

---

## 7. Testing

### test_analyzer.py
Test that known prompts produce the expected TaskCategory and RobotType:
- "Train a Franka to reach targets" → manipulation_reach, franka_panda
- "Balance a pole" → classic_cartpole, cartpole
- "Walk forward on flat ground" → locomotion_flat, anymal_d
- "Navigate rough terrain" → locomotion_rough, anymal_d

### test_generator.py
Test that the generator output:
- Contains `### FILE:` markers
- env_cfg parses as valid Python (ast.parse succeeds)
- Contains a class with "EnvCfg" in the name
- Contains a class with "RewardsCfg" in the name

### test_validator.py
Test the validator with:
- Valid Python with all required classes → is_valid=True
- Python with SyntaxError → is_valid=False, error message includes line number
- Python missing EnvCfg class → is_valid=False
- Python with unusually large reward weight → warning

---

## 8. Build Order (for Claude Code)

Execute in this exact sequence:

```
Phase 1: Skeleton
  1. Create all directories and empty __init__.py files
  2. Create pyproject.toml, .env.example, .gitignore, LICENSE
  3. Create all 4 prompt files with full content

Phase 2: Nemotron Client
  4. Implement robospec/nemotron/client.py
  5. Write a quick test: call Nemotron with "say hello" to verify API works

Phase 3: Knowledge Base
  6. Clone Isaac Lab to /tmp, extract example configs
  7. Extract MDP API reference (write extraction script)
  8. Create robots.json and reward_patterns.md

Phase 4: Pipeline
  9. Implement analyzer.py + test
  10. Implement context.py
  11. Implement generator.py + test
  12. Implement validator.py + test
  13. Implement explainer.py

Phase 5: CLI
  14. Implement cli.py with Typer
  15. End-to-end test: robospec generate "Balance a pole on a cart"
  16. Test all 4 task categories

Phase 6: Polish
  17. Streamlit UI (if time permits)
  18. README.md with usage examples
  19. Final end-to-end test of all components
```
