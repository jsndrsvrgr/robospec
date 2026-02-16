# RoboSpec

Natural language to NVIDIA Isaac Lab reinforcement learning environments.

**RoboSpec** takes a plain English description of a robot learning task and generates a complete, runnable Isaac Lab environment configuration — ready to train on a GPU.

> Built for the NVIDIA GTC 2026 Golden Ticket Developer Contest.

### CLI Tool

https://github.com/user-attachments/assets/45293903-5a7c-41fd-b404-2a8bfc19b4f5


## How It Works

```
"Train a Franka arm to reach random targets smoothly"
        │
        ▼
┌─────────────────────────────────────────┐
│  1. Analyze  — NL → structured TaskSpec │
│  2. Context  — select relevant examples │
│  3. Generate — Nemotron → config files  │
│  4. Validate — AST + semantic checks    │
│  5. Explain  — reward design writeup    │
└─────────────────────────────────────────┘
        │
        ▼
output/franka_panda_reach/
  ├── franka_panda_reach_env_cfg.py  # Full ManagerBasedRLEnvCfg
  ├── __init__.py                    # Gymnasium registration
  ├── train.py                       # Standalone training script
  └── README.md                      # Reward design explanation
```

## NVIDIA Technologies

| Technology | Role |
|-----------|------|
| **Nemotron** (`nvidia/llama-3.3-nemotron-super-49b-v1`) | LLM that generates all configs via NVIDIA NIM API |
| **Isaac Lab** | Target RL framework — generated configs run directly in it |
| **NVIDIA NIM** | API endpoint for Nemotron inference |

## Quick Start

### 1. Install

```bash
pip install -e ".[ui]"
```

### 2. Configure API keys

Copy `.env.example` to `.env` and add at least one key:

```bash
cp .env.example .env
# Edit .env with your NVIDIA_API_KEY and/or OPENROUTER_API_KEY
```

### 3. Generate

```bash
# Manipulation
robospec generate "Train a Franka arm to reach random targets smoothly"

# Classic control
robospec generate "Balance a pole on a cart"

# Locomotion (flat)
robospec generate "Walk forward on flat ground"

# Locomotion (rough)
robospec generate "Navigate rough terrain"

# With options
robospec generate "Reach targets precisely" --output ./my_task/ --verbose
```

### 4. Train (on an Isaac Lab machine)

```bash
scp -r output/franka_panda_reach/ user@gpu-machine:~/IsaacLab/
ssh user@gpu-machine
cd ~/IsaacLab && ./isaaclab.sh -p franka_panda_reach/train.py --headless
```

## Supported Tasks

| Category | Robot | Example Prompt |
|----------|-------|----------------|
| Manipulation (Reach) | Franka Panda | "Train a Franka to reach targets" |
| Classic Control | Cartpole | "Balance a pole on a cart" |
| Locomotion (Flat) | ANYmal-D | "Walk forward on flat ground" |
| Locomotion (Rough) | ANYmal-D | "Navigate rough terrain" |

## Streamlit Demo UI

```bash
streamlit run streamlit_app.py
```

## Architecture

```
robospec/
├── cli.py                  # Typer CLI entry point
├── pipeline/
│   ├── analyzer.py         # NL → TaskSpec (Nemotron)
│   ├── context.py          # Knowledge base selection
│   ├── generator.py        # TaskSpec → config files (Nemotron)
│   ├── validator.py        # AST + semantic validation
│   └── explainer.py        # Reward explanation (Nemotron)
├── nemotron/
│   └── client.py           # Async API client (NIM + OpenRouter fallback)
├── prompts/                # Prompt templates (.txt)
├── knowledge/              # Isaac Lab reference material
│   ├── examples/           # Real working env configs
│   ├── api_reference/      # MDP function signatures
│   ├── robots.json         # Robot catalog
│   └── reward_patterns.md  # Reward engineering heuristics
└── templates/              # Jinja2 templates for boilerplate
```

## Testing

```bash
pytest tests/ -v
```

## License

Apache 2.0
