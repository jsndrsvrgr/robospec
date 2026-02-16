"""Generate Isaac Lab config files from a TaskSpec using Nemotron."""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from robospec.nemotron.client import NemotronClient
from robospec.pipeline.analyzer import TaskSpec

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Map task categories to RoboSpec task ID naming (RoboSpec- prefix avoids
# collisions with built-in Isaac- tasks)
TASK_ID_MAP: dict[str, str] = {
    "manipulation_reach": "RoboSpec-Reach-{robot}-v0",
    "classic_cartpole": "RoboSpec-Cartpole-v0",
    "locomotion_flat": "RoboSpec-Velocity-Flat-{robot}-v0",
    "locomotion_rough": "RoboSpec-Velocity-Rough-{robot}-v0",
}

# Map robot types to pretty names used in task IDs
ROBOT_ID_MAP: dict[str, str] = {
    "franka_panda": "Franka",
    "cartpole": "Cartpole",
    "anymal_d": "Anymal-D",
}

# Map categories to the best RL framework for training
TRAIN_FRAMEWORK_MAP: dict[str, str] = {
    "manipulation_reach": "skrl",
    "classic_cartpole": "rl_games",
    "locomotion_flat": "rsl_rl",
    "locomotion_rough": "rsl_rl",
}

# Per-category training configuration. Agent config paths are verified against
# real Isaac Lab source (isaaclab_tasks/manager_based/*/agents/).
@dataclass
class CategoryTrainConfig:
    framework: str
    default_max_iterations: int
    rl_games_agent_cfg: str  # module:yaml or empty
    rsl_rl_agent_cfg: str    # module.submodule:ClassName or empty
    skrl_agent_cfg: str      # module:yaml or empty

CATEGORY_TRAIN_CONFIG: dict[str, CategoryTrainConfig] = {
    "classic_cartpole": CategoryTrainConfig(
        framework="rl_games",
        default_max_iterations=200,
        rl_games_agent_cfg="isaaclab_tasks.manager_based.classic.cartpole.agents:rl_games_ppo_cfg.yaml",
        rsl_rl_agent_cfg="isaaclab_tasks.manager_based.classic.cartpole.agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        skrl_agent_cfg="isaaclab_tasks.manager_based.classic.cartpole.agents:skrl_ppo_cfg.yaml",
    ),
    "manipulation_reach": CategoryTrainConfig(
        framework="skrl",
        default_max_iterations=500,
        rl_games_agent_cfg="isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents:rl_games_ppo_cfg.yaml",
        rsl_rl_agent_cfg="isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        skrl_agent_cfg="isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents:skrl_ppo_cfg.yaml",
    ),
    "locomotion_flat": CategoryTrainConfig(
        framework="rsl_rl",
        default_max_iterations=1500,
        rl_games_agent_cfg="",
        rsl_rl_agent_cfg="isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
        skrl_agent_cfg="isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents:skrl_flat_ppo_cfg.yaml",
    ),
    "locomotion_rough": CategoryTrainConfig(
        framework="rsl_rl",
        default_max_iterations=3000,
        rl_games_agent_cfg="",
        rsl_rl_agent_cfg="isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg:AnymalDRoughPPORunnerCfg",
        skrl_agent_cfg="isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents:skrl_rough_ppo_cfg.yaml",
    ),
}

# Category-specific approved MDP functions. Every name here has been verified
# against either the API reference markdown or the real example configs.
CATEGORY_APPROVED_FUNCTIONS: dict[str, dict[str, list[str]]] = {
    "manipulation_reach": {
        "rewards": [
            "position_command_error", "position_command_error_tanh",
            "orientation_command_error", "action_rate_l2", "joint_vel_l2",
            "joint_acc_l2", "is_terminated",
        ],
        "observations": [
            "joint_pos_rel", "joint_vel_rel", "generated_commands", "last_action",
        ],
        "terminations": ["time_out"],
        "events": ["reset_joints_by_scale", "reset_scene_to_default"],
        "actions": ["JointPositionActionCfg", "DifferentialInverseKinematicsActionCfg"],
        "commands": ["UniformPoseCommandCfg"],
    },
    "classic_cartpole": {
        "rewards": [
            "is_alive", "is_terminated", "joint_pos_target_l2",
            "joint_vel_l1", "joint_vel_l2", "action_l2",
        ],
        "observations": ["joint_pos_rel", "joint_vel_rel"],
        "terminations": ["time_out", "joint_pos_out_of_manual_limit"],
        "events": ["reset_joints_by_offset"],
        "actions": ["JointEffortActionCfg"],
        "commands": [],
    },
    "locomotion_flat": {
        "rewards": [
            "track_lin_vel_xy_exp", "track_ang_vel_z_exp",
            "lin_vel_z_l2", "ang_vel_xy_l2", "flat_orientation_l2",
            "joint_torques_l2", "action_rate_l2", "joint_acc_l2",
            "feet_air_time", "undesired_contacts", "is_terminated",
            "joint_pos_limits",
        ],
        "observations": [
            "base_lin_vel", "base_ang_vel", "projected_gravity",
            "joint_pos_rel", "joint_vel_rel", "generated_commands", "last_action",
        ],
        "terminations": ["time_out", "illegal_contact"],
        "events": [
            "reset_root_state_uniform", "reset_joints_by_scale",
            "push_by_setting_velocity", "apply_external_force_torque",
            "randomize_rigid_body_material", "randomize_rigid_body_mass",
            "randomize_rigid_body_com",
        ],
        "actions": ["JointPositionActionCfg"],
        "commands": ["UniformVelocityCommandCfg"],
    },
    "locomotion_rough": {
        "rewards": [
            "track_lin_vel_xy_exp", "track_ang_vel_z_exp",
            "lin_vel_z_l2", "ang_vel_xy_l2", "flat_orientation_l2",
            "joint_torques_l2", "action_rate_l2", "joint_acc_l2",
            "feet_air_time", "undesired_contacts", "is_terminated",
            "joint_pos_limits",
        ],
        "observations": [
            "base_lin_vel", "base_ang_vel", "projected_gravity",
            "joint_pos_rel", "joint_vel_rel", "generated_commands",
            "last_action", "height_scan",
        ],
        "terminations": ["time_out", "illegal_contact"],
        "events": [
            "reset_root_state_uniform", "reset_joints_by_scale",
            "push_by_setting_velocity", "apply_external_force_torque",
            "randomize_rigid_body_material", "randomize_rigid_body_mass",
            "randomize_rigid_body_com",
        ],
        "actions": ["JointPositionActionCfg"],
        "commands": ["UniformVelocityCommandCfg"],
    },
}


# Per-category robot config: the correct import line and robot assignment.
# Verified against real Isaac Lab example configs in knowledge/examples/.
@dataclass
class RobotConfig:
    import_line: str    # e.g. "from isaaclab_assets.robots.cartpole import CARTPOLE_CFG"
    cfg_name: str       # e.g. "CARTPOLE_CFG"
    robot_line: str     # e.g. 'CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")'

CATEGORY_ROBOT_CONFIG: dict[str, RobotConfig] = {
    "classic_cartpole": RobotConfig(
        import_line="from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip",
        cfg_name="CARTPOLE_CFG",
        robot_line='CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
    ),
    "manipulation_reach": RobotConfig(
        import_line="from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip",
        cfg_name="FRANKA_PANDA_HIGH_PD_CFG",
        robot_line='FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
    ),
    "locomotion_flat": RobotConfig(
        import_line="from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip",
        cfg_name="ANYMAL_D_CFG",
        robot_line='ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
    ),
    "locomotion_rough": RobotConfig(
        import_line="from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip",
        cfg_name="ANYMAL_D_CFG",
        robot_line='ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
    ),
}


def post_process_env_cfg(code: str, category: str) -> tuple[str, list[str]]:
    """Post-process generated env_cfg to fix robot imports and mdp imports.

    Fixes:
    1. Replace task-specific mdp imports with core isaaclab.envs.mdp
    2. Ensure pre-built robot config is imported (not defined inline)
    3. Replace inline robot ArticulationCfg definitions with .replace() pattern

    Returns (fixed_code, list_of_fixes_applied).
    """
    fixes: list[str] = []
    result = code

    # 1. Fix mdp imports: replace task-specific with core
    # Pattern A: "import isaaclab_tasks.manager_based.*.mdp as mdp"
    task_mdp_pattern = re.compile(
        r"import\s+isaaclab_tasks\.manager_based\.\S+\.mdp\s+as\s+mdp"
    )
    if task_mdp_pattern.search(result):
        result = task_mdp_pattern.sub("import isaaclab.envs.mdp as mdp", result)
        fixes.append("Replaced task-specific mdp import with isaaclab.envs.mdp")

    # Pattern B: "from isaaclab_tasks.manager_based.*.mdp import (...)" (multi-line)
    # These import classes like DifferentialInverseKinematicsActionCfg, UniformPoseCommandCfg, mdp
    # which are all available via isaaclab.envs.mdp — replace entire block.
    task_mdp_from_pattern = re.compile(
        r"from\s+isaaclab_tasks\.manager_based\.\S+\.mdp\s+import\s+\([^)]*\)\s*\n?",
        re.DOTALL,
    )
    if task_mdp_from_pattern.search(result):
        # Check if "import isaaclab.envs.mdp as mdp" already exists
        if "import isaaclab.envs.mdp as mdp" not in result:
            result = task_mdp_from_pattern.sub("import isaaclab.envs.mdp as mdp\n", result)
        else:
            result = task_mdp_from_pattern.sub("", result)
        fixes.append("Replaced task-specific from-import mdp block with isaaclab.envs.mdp")

    # Pattern C: single-line "from isaaclab_tasks.manager_based.*.mdp import X, Y"
    task_mdp_from_single = re.compile(
        r"from\s+isaaclab_tasks\.manager_based\.\S+\.mdp\s+import\s+[^\n(]+\n?"
    )
    if task_mdp_from_single.search(result):
        if "import isaaclab.envs.mdp as mdp" not in result:
            result = task_mdp_from_single.sub("import isaaclab.envs.mdp as mdp\n", result)
        else:
            result = task_mdp_from_single.sub("", result)
        fixes.append("Replaced task-specific from-import mdp line with isaaclab.envs.mdp")

    # 2. Ensure robot config import exists
    robot_cfg = CATEGORY_ROBOT_CONFIG.get(category)
    if robot_cfg and robot_cfg.cfg_name not in result:
        # Find a safe injection point: after the last single-line isaaclab/isaaclab_assets
        # import (exclude isaaclab_tasks, and skip multi-line imports with open parens)
        import_section_end = 0
        for match in re.finditer(
            r"^(?:from|import)\s+isaaclab(?!_tasks)\S*[^(\n]*$", result, re.MULTILINE
        ):
            import_section_end = match.end()

        if import_section_end > 0:
            result = (
                result[:import_section_end]
                + f"\n\n{robot_cfg.import_line}"
                + result[import_section_end:]
            )
            fixes.append(f"Injected robot config import: {robot_cfg.cfg_name}")

    # 3. Replace inline robot definitions with .replace() pattern
    if robot_cfg:
        # Match patterns like:
        #   robot: ArticulationCfg = AssetBaseCfg(...)  (multi-line)
        #   robot = ArticulationCfg(...)  (multi-line)
        #   robot: ArticulationCfg = MISSING
        # But NOT already correct patterns like CARTPOLE_CFG.replace(...)
        inline_robot_pattern = re.compile(
            r"(    robot\s*(?::\s*ArticulationCfg\s*)?=\s*)"
            r"(?!.*\.replace\().*$",
            re.MULTILINE,
        )
        match = inline_robot_pattern.search(result)
        if match and robot_cfg.cfg_name not in match.group(0):
            # Find the full extent of the inline definition (may span multiple lines)
            start = match.start()
            # Find the end: look for the next class attribute at the same indent level
            # or end of the class body
            rest = result[match.end():]
            # Count parentheses to find the end of multi-line definitions
            paren_depth = result[start:match.end()].count("(") - result[start:match.end()].count(")")
            end = match.end()
            for i, char in enumerate(rest):
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                if paren_depth <= 0:
                    # Find the end of this line
                    nl = rest.find("\n", i)
                    end = match.end() + (nl if nl >= 0 else len(rest))
                    break

            indent = match.group(1).split("robot")[0]
            replacement = f"{indent}robot: ArticulationCfg = {robot_cfg.robot_line}"
            result = result[:start] + replacement + result[end:]
            fixes.append(f"Replaced inline robot definition with {robot_cfg.cfg_name}.replace()")

    # 4. Remove literal "ISAACLAB_NUCLEUS_DIR" string references (not variable refs)
    # These are strings like "ISAACLAB_NUCLEUS_DIR/Robots/..." that should be variable refs
    literal_nucleus = re.compile(r'"ISAACLAB_NUCLEUS_DIR/[^"]*"')
    if literal_nucleus.search(result):
        # This means there's a string literal with the path — the robot replacement
        # above should have already fixed this, but clean up any remaining ones
        fixes.append("Warning: literal ISAACLAB_NUCLEUS_DIR string found in code")

    return result, fixes


def _format_approved_functions(category: str) -> str:
    """Format category-specific approved functions for the generation prompt."""
    funcs = CATEGORY_APPROVED_FUNCTIONS.get(category, {})
    if not funcs:
        return ""
    lines = ["APPROVED FUNCTIONS FOR THIS TASK (use ONLY these):"]
    for section, names in funcs.items():
        if names:
            lines.append(f"  {section.title()}: {', '.join(f'mdp.{n}' for n in names)}")
    return "\n".join(lines)


@dataclass
class GeneratedConfig:
    env_cfg: str  # Main environment config Python code
    init_py: str  # Gymnasium registration with agent config entry points
    train_script: str  # Standalone train.py script
    readme: str  # Explanation (generated later by explainer)
    raw_response: str  # Full Nemotron response for debugging
    task_name: str  # Sanitized module name, e.g., "franka_panda_reach"
    task_id: str  # RoboSpec gym ID, e.g., "RoboSpec-Reach-Franka-v0"


def sanitize_module_name(name: str) -> str:
    """Sanitize a string into a valid Python module name.

    - Replaces hyphens and spaces with underscores
    - Lowercases everything
    - Strips any character that isn't [a-z0-9_]
    - Prepends underscore if starts with digit
    """
    name = name.lower()
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    if name and name[0].isdigit():
        name = f"_{name}"
    return name


def _make_task_name(task_spec: TaskSpec) -> str:
    """Derive a sanitized task name from the spec."""
    robot = task_spec.robot.value
    category = task_spec.category.value.split("_", 1)[-1]
    return sanitize_module_name(f"{robot}_{category}")


def _make_task_id(task_spec: TaskSpec) -> str:
    """Derive the RoboSpec gymnasium task ID."""
    pattern = TASK_ID_MAP.get(task_spec.category.value, "RoboSpec-Custom-v0")
    robot_pretty = ROBOT_ID_MAP.get(task_spec.robot.value, "Robot")
    return pattern.format(robot=robot_pretty)


def _find_env_cfg_class(code: str) -> str | None:
    """Extract the name of the main EnvCfg class from generated code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "EnvCfg" in node.name:
            return node.name
    return None


def _generate_init_py(
    task_name: str, task_id: str, env_cfg_class: str, train_cfg: CategoryTrainConfig | None,
) -> str:
    """Generate __init__.py with gym.register() and agent config entry points."""
    cfg_module = f"{task_name}_env_cfg"

    # Build kwargs dict entries
    kwargs_lines = [
        f'        "env_cfg_entry_point": f"{{__name__}}.{cfg_module}:{env_cfg_class}",',
    ]
    if train_cfg:
        if train_cfg.rl_games_agent_cfg:
            kwargs_lines.append(
                f'        "rl_games_cfg_entry_point": "{train_cfg.rl_games_agent_cfg}",'
            )
        if train_cfg.rsl_rl_agent_cfg:
            kwargs_lines.append(
                f'        "rsl_rl_cfg_entry_point": "{train_cfg.rsl_rl_agent_cfg}",'
            )
        if train_cfg.skrl_agent_cfg:
            kwargs_lines.append(
                f'        "skrl_cfg_entry_point": "{train_cfg.skrl_agent_cfg}",'
            )

    kwargs_block = "\n".join(kwargs_lines)

    return f'''"""Registration for {task_id}."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="{task_id}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={{
{kwargs_block}
    }},
)
'''


def _generate_train_py(
    task_name: str,
    task_id: str,
    env_cfg_class: str,
    num_envs: int,
    train_cfg: CategoryTrainConfig,
) -> str:
    """Generate a standalone train.py using the Jinja2 template."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)
    template = env.get_template("train.py.j2")

    cfg_module = f"{task_name}_env_cfg"

    return template.render(
        task_id=task_id,
        default_num_envs=num_envs,
        default_max_iterations=train_cfg.default_max_iterations,
        env_cfg_module=cfg_module,
        env_cfg_class=env_cfg_class,
        rl_games_agent_cfg=train_cfg.rl_games_agent_cfg,
        rsl_rl_agent_cfg=train_cfg.rsl_rl_agent_cfg,
        skrl_agent_cfg=train_cfg.skrl_agent_cfg,
        rl_framework=train_cfg.framework,
    )


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences from code blocks."""
    code = re.sub(r"^```(?:python|bash|sh)?\s*\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)
    return code.strip()


def _parse_response(response: str, task_name: str) -> dict[str, str]:
    """Parse Nemotron's multi-file response into {filename: content} dict.

    Expected format:
    ### FILE: filename.py
    [code]
    """
    files: dict[str, str] = {}

    # Split on ### FILE: markers
    parts = re.split(r"###\s*FILE:\s*", response)

    if len(parts) <= 1:
        # No markers found — treat entire response as env_cfg
        return {f"{task_name}_env_cfg.py": _strip_code_fences(response)}

    for part in parts[1:]:  # Skip text before first marker
        lines = part.strip().split("\n", 1)
        if not lines:
            continue
        filename = lines[0].strip().rstrip(":")
        content = lines[1] if len(lines) > 1 else ""
        files[filename] = _strip_code_fences(content)

    return files


async def generate_config(
    client: NemotronClient,
    task_spec: TaskSpec,
    context: str,
) -> GeneratedConfig:
    """Generate Isaac Lab configuration files.

    Nemotron generates only the env_cfg.py. The __init__.py and train.sh
    are produced deterministically via post-processing to ensure correctness.
    """
    system_prompt = (PROMPTS_DIR / "system.txt").read_text()
    generate_template = (PROMPTS_DIR / "generate.txt").read_text()

    task_name = _make_task_name(task_spec)
    task_id = _make_task_id(task_spec)
    framework = TRAIN_FRAMEWORK_MAP.get(task_spec.category.value, "skrl")

    approved_funcs = _format_approved_functions(task_spec.category.value)

    user_prompt = generate_template.format(
        category=task_spec.category.value,
        robot=task_spec.robot.value,
        objectives=", ".join(task_spec.objectives),
        constraints=", ".join(task_spec.constraints),
        difficulty=task_spec.difficulty,
        episode_length_s=task_spec.episode_length_s,
        num_envs=task_spec.num_envs,
        task_name=task_name,
        approved_functions=approved_funcs,
    )

    # Prepend context to system prompt
    full_system = system_prompt + "\n\n" + context

    response = await client.generate(
        system_prompt=full_system,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=8192,
    )

    env_cfg_code = _extract_env_cfg(response, task_name)
    return _build_config(env_cfg_code, response, task_name, task_id, task_spec.num_envs, task_spec.category.value)


async def repair_config(
    client: NemotronClient,
    original_code: str,
    errors: list[str],
    context: str,
    whitelist_hint: str = "",
) -> str:
    """Repair a generated env_cfg using the repair prompt.

    Returns the repaired env_cfg code string.
    """
    system_prompt = (PROMPTS_DIR / "system.txt").read_text()
    repair_template = (PROMPTS_DIR / "repair.txt").read_text()

    user_prompt = repair_template.replace(
        "{error_list}", "\n".join(f"- {e}" for e in errors)
    ).replace(
        "{whitelist_hint}", whitelist_hint
    ).replace(
        "{original_code}", original_code
    )

    full_system = system_prompt + "\n\n" + context

    response = await client.generate(
        system_prompt=full_system,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=8192,
    )

    return _strip_code_fences(response)


def _extract_env_cfg(response: str, task_name: str) -> str:
    """Extract env_cfg code from Nemotron response."""
    files = _parse_response(response, task_name)
    env_cfg_key = next(
        (k for k in files if "env_cfg" in k.lower()), None
    )
    return files.get(env_cfg_key, "") if env_cfg_key else _strip_code_fences(response)


def _build_config(
    env_cfg_code: str,
    raw_response: str,
    task_name: str,
    task_id: str,
    num_envs: int,
    category: str,
) -> GeneratedConfig:
    """Build a GeneratedConfig with deterministic __init__.py and train.py."""
    # Post-process the env_cfg to fix robot imports and mdp imports
    env_cfg_code, _pp_fixes = post_process_env_cfg(env_cfg_code, category)

    env_cfg_class = _find_env_cfg_class(env_cfg_code) or "EnvCfg"
    train_cfg = CATEGORY_TRAIN_CONFIG.get(category)

    init_py = _generate_init_py(task_name, task_id, env_cfg_class, train_cfg)
    train_script = _generate_train_py(
        task_name, task_id, env_cfg_class, num_envs, train_cfg,
    ) if train_cfg else ""

    return GeneratedConfig(
        env_cfg=env_cfg_code,
        init_py=init_py,
        train_script=train_script,
        readme="",
        raw_response=raw_response,
        task_name=task_name,
        task_id=task_id,
    )
