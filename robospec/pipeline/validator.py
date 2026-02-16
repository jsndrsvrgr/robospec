"""Validate generated Isaac Lab configuration code."""

import ast
import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

# Functions/classes used in our example configs that come from task-specific
# mdp modules (e.g. isaaclab_tasks.manager_based.locomotion.velocity.mdp)
# rather than the core isaaclab.envs.mdp. These are legitimate Isaac Lab
# symbols that our API reference extraction didn't capture.
SUPPLEMENTAL_WHITELIST: set[str] = {
    # Reach task rewards (from reach.mdp)
    "position_command_error",
    "position_command_error_tanh",
    "orientation_command_error",
    # Cartpole task rewards (from classic.cartpole.mdp)
    "joint_pos_target_l2",
    # Locomotion task rewards (from locomotion.velocity.mdp)
    "feet_air_time",
    # Command configurations (accessed as mdp.XxxCommandCfg in examples)
    "UniformVelocityCommandCfg",
    "UniformPoseCommandCfg",
    "NullCommandCfg",
    # Locomotion event functions (from locomotion.velocity.mdp)
    "randomize_rigid_body_material",
    "randomize_rigid_body_mass",
    # Curriculum functions
    "modify_reward_weight",
    "terrain_levels_vel",
}

# Deterministic text-replacement map for common LLM hallucinations.
# key = hallucinated name, value = correct replacement (or None to skip).
COMMON_CORRECTIONS: dict[str, str | None] = {
    # Near-miss reward functions
    "joint_pos_l2": "joint_pos_target_l2",
    "joint_pos_l1": "joint_deviation_l1",
    "joint_vel_l2_asset": "joint_vel_l2",
    "action_rate_l2_norm": "action_rate_l2",
    "track_lin_vel_xy": "track_lin_vel_xy_exp",
    "track_ang_vel_z": "track_ang_vel_z_exp",
    "base_lin_vel_z_l2": "lin_vel_z_l2",
    "base_ang_vel_xy_l2": "ang_vel_xy_l2",
    "position_error_tanh": "position_command_error_tanh",
    "position_error": "position_command_error",
    "joint_pos_target_l1": "joint_pos_target_l2",
    "action_rate_l1": "action_rate_l2",
    "feet_air_time_biped_reward": "feet_air_time",
}


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_api_whitelist() -> set[str]:
    """Load all known MDP function/class names from the API reference markdown.

    Scans all files in knowledge/api_reference/ for markdown headers:
    - "### mdp.function_name" -> extracts "function_name"
    - "### ClassName" -> extracts "ClassName"

    Also includes SUPPLEMENTAL_WHITELIST for task-specific functions used
    in our example configs but not in the core isaaclab.envs.mdp.

    Returns a cached set of allowed symbol names.
    """
    symbols: set[str] = set()
    api_dir = KNOWLEDGE_DIR / "api_reference"

    if not api_dir.exists():
        return symbols

    for md_file in api_dir.glob("*.md"):
        text = md_file.read_text()
        # Match "### mdp.function_name" pattern
        for match in re.finditer(r"^###\s+mdp\.(\w+)", text, re.MULTILINE):
            symbols.add(match.group(1))
        # Match "### ClassName" pattern (for action classes etc.)
        for match in re.finditer(r"^###\s+([A-Z]\w+)", text, re.MULTILINE):
            symbols.add(match.group(1))

    symbols.update(SUPPLEMENTAL_WHITELIST)
    return symbols


def auto_correct_code(code: str) -> tuple[str, list[str]]:
    """Apply deterministic text replacements for common LLM hallucinations.

    Uses word-boundary-aware regex to avoid corrupting valid names
    (e.g. mdp.track_lin_vel_xy must not match inside mdp.track_lin_vel_xy_exp).

    Returns (corrected_code, list_of_corrections_applied).
    This should be called BEFORE validation and the repair loop.
    """
    corrections: list[str] = []
    corrected = code

    for wrong, right in COMMON_CORRECTIONS.items():
        if right is None:
            continue
        # Use word boundary (\b) after the name to prevent substring matches
        pattern = re.compile(rf"mdp\.{re.escape(wrong)}\b")
        if pattern.search(corrected):
            corrected = pattern.sub(f"mdp.{right}", corrected)
            corrections.append(f"mdp.{wrong} -> mdp.{right}")

    return corrected, corrections


def check_api_symbols(code: str, whitelist: set[str]) -> list[str]:
    """Check that all mdp.X attribute accesses use whitelisted symbols.

    Walks the AST looking for patterns like `mdp.something` and returns
    error messages for any `something` not in the whitelist.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # Syntax errors are caught by the main validator

    unknown: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "mdp"
            and node.attr not in seen
        ):
            name = node.attr
            seen.add(name)
            if name not in whitelist:
                # Also allow nested access like mdp.UniformVelocityCommandCfg.Ranges
                # by checking if any whitelist entry is a prefix
                suggestion = ""
                matches = difflib.get_close_matches(name, whitelist, n=1, cutoff=0.6)
                if matches:
                    suggestion = f" Did you mean: mdp.{matches[0]}?"
                unknown.append(
                    f"Unknown MDP function: mdp.{name}.{suggestion}"
                )

    return unknown


def validate_config(code: str) -> ValidationResult:
    """Run all validation checks on generated env_cfg code.

    Checks:
    1. AST Parse — code must be valid Python
    2. Class Check — at least one class containing "EnvCfg"
    3. Reward Check — at least one class containing "RewardsCfg"
    4. Post-init Check — a __post_init__ method must exist
    5. Import Check — must import from isaaclab or omni.isaac
    6. API Whitelist — all mdp.X symbols must be known
    """
    result = ValidationResult()

    # 1. AST Parse
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result.is_valid = False
        result.errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        return result  # Can't check further if syntax is broken

    # Collect class names and method names
    class_names: list[str] = []
    method_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
        if isinstance(node, ast.FunctionDef):
            method_names.append(node.name)

    # 2. Class Check — EnvCfg
    has_env_cfg = any("EnvCfg" in name for name in class_names)
    if not has_env_cfg:
        result.is_valid = False
        result.errors.append(
            "Missing environment config class: no class with 'EnvCfg' in name found"
        )

    # 3. Reward Check — RewardsCfg
    has_rewards_cfg = any("RewardsCfg" in name for name in class_names)
    if not has_rewards_cfg:
        result.is_valid = False
        result.errors.append(
            "Missing rewards config: no class with 'RewardsCfg' in name found"
        )

    # 4. Post-init Check
    has_post_init = "__post_init__" in method_names
    if not has_post_init:
        result.is_valid = False
        result.errors.append("Missing __post_init__ method")

    # 5. Import Check
    has_isaac_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "isaaclab" in alias.name or "omni.isaac" in alias.name:
                    has_isaac_import = True
                    break
        elif isinstance(node, ast.ImportFrom):
            if node.module and ("isaaclab" in node.module or "omni.isaac" in node.module):
                has_isaac_import = True

    if not has_isaac_import:
        result.is_valid = False
        result.errors.append(
            "Missing Isaac Lab imports: no imports from 'isaaclab' or 'omni.isaac' found"
        )

    # 6. API Whitelist Check
    whitelist = load_api_whitelist()
    if whitelist:  # Only check if whitelist was loaded successfully
        unknown_symbols = check_api_symbols(code, whitelist)
        if unknown_symbols:
            result.is_valid = False
            result.errors.extend(unknown_symbols)

    # 7. Inline robot check — literal ISAACLAB_NUCLEUS_DIR strings
    if '"ISAACLAB_NUCLEUS_DIR/' in code or "'ISAACLAB_NUCLEUS_DIR/" in code:
        result.is_valid = False
        result.errors.append(
            "Literal ISAACLAB_NUCLEUS_DIR string found — use pre-built robot config "
            "(e.g. CARTPOLE_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')) instead"
        )

    # 8. Task-specific mdp import check
    if re.search(r"import\s+isaaclab_tasks\.manager_based\.\S+\.mdp\s+as\s+mdp", code):
        result.warnings.append(
            "Task-specific mdp import found (isaaclab_tasks.manager_based.*.mdp). "
            "Use 'import isaaclab.envs.mdp as mdp' for external configs."
        )

    # Warnings: check for unusually large reward weights
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "weight":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                weight = abs(node.value.value)
                if weight > 10.0:
                    result.warnings.append(
                        f"Unusually large reward weight: {node.value.value}"
                    )
            elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub):
                if isinstance(node.value.operand, ast.Constant):
                    weight = abs(node.value.operand.value)
                    if weight > 10.0:
                        result.warnings.append(
                            f"Unusually large reward weight: -{node.value.operand.value}"
                        )

    return result
