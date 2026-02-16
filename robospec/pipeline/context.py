"""Context Builder — Deterministic Retrieval Strategy

We use a hand-curated EXAMPLE_MAP to select which Isaac Lab examples and
API references to include in the generation prompt. This is intentional:

- At ~104K tokens of total knowledge base, everything fits in context
- Deterministic selection is debuggable: you know exactly what the model saw
- No embedding model, vector store, or chunking logic to fail silently
- RAG should only be added when knowledge base exceeds ~500K tokens

The EXAMPLE_MAP should be expanded (not replaced) as new robot/task
categories are added.
"""

from pathlib import Path

from robospec.pipeline.analyzer import TaskSpec

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

# Map task categories to relevant example files (includes base configs)
EXAMPLE_MAP: dict[str, list[str]] = {
    "manipulation_reach": [
        "reach_env_cfg_base.py",
        "franka_reach_env_cfg.py",
        "franka_reach_joint_pos_env_cfg.py",
        "cartpole_env_cfg.py",
    ],
    "classic_cartpole": [
        "cartpole_env_cfg.py",
        "reach_env_cfg_base.py",
        "franka_reach_env_cfg.py",
    ],
    "locomotion_flat": [
        "velocity_env_cfg_base.py",
        "anymal_d_flat_env_cfg.py",
        "anymal_d_rough_env_cfg.py",
    ],
    "locomotion_rough": [
        "velocity_env_cfg_base.py",
        "anymal_d_rough_env_cfg.py",
        "anymal_d_flat_env_cfg.py",
    ],
}

# API reference files to always include
API_REFERENCE_FILES = [
    "mdp_rewards.md",
    "mdp_observations.md",
    "mdp_actions.md",
    "mdp_terminations.md",
    "mdp_events.md",
]


def build_context(task_spec: TaskSpec) -> str:
    """Assemble knowledge base context for the generation prompt.

    Always includes all API reference files, robots.json, and reward_patterns.md.
    Conditionally includes example configs based on the task category.
    """
    sections: list[str] = []

    # 1. API Reference
    sections.append("=== ISAAC LAB API REFERENCE ===\n")
    api_dir = KNOWLEDGE_DIR / "api_reference"
    for filename in API_REFERENCE_FILES:
        filepath = api_dir / filename
        if filepath.exists():
            sections.append(f"--- {filename} ---")
            sections.append(filepath.read_text())
            sections.append("")

    # 2. Robot Specifications
    robots_path = KNOWLEDGE_DIR / "robots.json"
    if robots_path.exists():
        sections.append("=== ROBOT SPECIFICATIONS ===\n")
        sections.append(robots_path.read_text())
        sections.append("")

    # 3. Reward Patterns
    patterns_path = KNOWLEDGE_DIR / "reward_patterns.md"
    if patterns_path.exists():
        sections.append("=== REWARD ENGINEERING PATTERNS ===\n")
        sections.append(patterns_path.read_text())
        sections.append("")

    # 4. Working Example Configurations
    category_key = task_spec.category.value
    example_files = EXAMPLE_MAP.get(category_key, [])

    if example_files:
        sections.append("=== WORKING EXAMPLE CONFIGURATIONS ===")
        sections.append(
            "Follow these patterns exactly. These are real, working Isaac Lab configs.\n"
        )

        examples_dir = KNOWLEDGE_DIR / "examples"
        for filename in example_files:
            filepath = examples_dir / filename
            if filepath.exists():
                sections.append(f"--- {filename} ---")
                sections.append("```python")
                sections.append(filepath.read_text())
                sections.append("```\n")

    return "\n".join(sections)
