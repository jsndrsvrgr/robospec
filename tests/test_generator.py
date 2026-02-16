"""Tests for the config generator's parsing and post-processing logic."""

import ast
import pytest

from robospec.pipeline.generator import (
    _parse_response,
    _strip_code_fences,
    _make_task_name,
    _make_task_id,
    _find_env_cfg_class,
    _generate_init_py,
    _generate_train_py,
    sanitize_module_name,
    post_process_env_cfg,
    CATEGORY_TRAIN_CONFIG,
)
from robospec.pipeline.analyzer import TaskSpec, TaskCategory, RobotType


class TestSanitizeModuleName:
    def test_replaces_hyphens(self):
        assert sanitize_module_name("franka-panda-reach") == "franka_panda_reach"

    def test_replaces_spaces(self):
        assert sanitize_module_name("my task name") == "my_task_name"

    def test_lowercases(self):
        assert sanitize_module_name("FrankaReach") == "frankareach"

    def test_strips_special_chars(self):
        assert sanitize_module_name("task@#$name!") == "taskname"

    def test_prepends_underscore_for_digit_start(self):
        assert sanitize_module_name("3d_task") == "_3d_task"

    def test_mixed(self):
        assert sanitize_module_name("My-Task 2!") == "my_task_2"


class TestStripCodeFences:
    def test_python_fences(self):
        code = '```python\nprint("hello")\n```'
        assert _strip_code_fences(code) == 'print("hello")'

    def test_no_fences(self):
        code = 'print("hello")'
        assert _strip_code_fences(code) == 'print("hello")'

    def test_bash_fences(self):
        code = "```bash\necho hi\n```"
        assert _strip_code_fences(code) == "echo hi"


class TestParseResponse:
    def test_multi_file_response(self):
        response = (
            "### FILE: reach_env_cfg.py\n"
            "```python\nimport isaaclab\nclass ReachEnvCfg:\n    pass\n```\n\n"
            "### FILE: __init__.py\n"
            "import gym\n\n"
            "### FILE: train.sh\n"
            "```bash\npython train.py\n```"
        )
        files = _parse_response(response, "franka_reach")
        assert "reach_env_cfg.py" in files
        assert "import isaaclab" in files["reach_env_cfg.py"]

    def test_no_markers_fallback(self):
        response = "import isaaclab\nclass MyEnvCfg:\n    pass"
        files = _parse_response(response, "franka_reach")
        assert "franka_reach_env_cfg.py" in files
        assert "import isaaclab" in files["franka_reach_env_cfg.py"]

    def test_env_cfg_parses_as_valid_python(self):
        response = (
            "### FILE: test_env_cfg.py\n"
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "\nclass TestEnvCfg(ManagerBasedRLEnvCfg):\n"
            "    pass\n\n"
        )
        files = _parse_response(response, "test")
        env_cfg = files["test_env_cfg.py"]
        tree = ast.parse(env_cfg)
        assert tree is not None

    def test_contains_env_cfg_class(self):
        response = (
            "### FILE: test_env_cfg.py\n"
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "\nclass TestEnvCfg(ManagerBasedRLEnvCfg):\n"
            "    pass\n"
        )
        files = _parse_response(response, "test")
        env_cfg = files["test_env_cfg.py"]
        tree = ast.parse(env_cfg)
        class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert any("EnvCfg" in name for name in class_names)


class TestMakeTaskName:
    def test_franka_reach(self):
        spec = TaskSpec(
            category=TaskCategory.MANIPULATION_REACH,
            robot=RobotType.FRANKA_PANDA,
            description="test",
            objectives=[],
            constraints=[],
        )
        assert _make_task_name(spec) == "franka_panda_reach"

    def test_cartpole(self):
        spec = TaskSpec(
            category=TaskCategory.CLASSIC_CARTPOLE,
            robot=RobotType.CARTPOLE,
            description="test",
            objectives=[],
            constraints=[],
        )
        assert _make_task_name(spec) == "cartpole_cartpole"

    def test_anymal_locomotion(self):
        spec = TaskSpec(
            category=TaskCategory.LOCOMOTION_FLAT,
            robot=RobotType.ANYMAL_D,
            description="test",
            objectives=[],
            constraints=[],
        )
        assert _make_task_name(spec) == "anymal_d_flat"

    def test_anymal_rough(self):
        spec = TaskSpec(
            category=TaskCategory.LOCOMOTION_ROUGH,
            robot=RobotType.ANYMAL_D,
            description="test",
            objectives=[],
            constraints=[],
        )
        assert _make_task_name(spec) == "anymal_d_rough"

    def test_no_hyphens_in_any_name(self):
        """All task names must be valid Python module names (no hyphens)."""
        for cat in TaskCategory:
            for robot in RobotType:
                spec = TaskSpec(
                    category=cat, robot=robot,
                    description="test", objectives=[], constraints=[],
                )
                name = _make_task_name(spec)
                assert "-" not in name, f"Hyphen found in task name: {name}"
                assert name.isidentifier(), f"Not a valid Python identifier: {name}"


class TestMakeTaskId:
    def test_franka_reach_id(self):
        spec = TaskSpec(
            category=TaskCategory.MANIPULATION_REACH,
            robot=RobotType.FRANKA_PANDA,
            description="test", objectives=[], constraints=[],
        )
        assert _make_task_id(spec) == "RoboSpec-Reach-Franka-v0"

    def test_cartpole_id(self):
        spec = TaskSpec(
            category=TaskCategory.CLASSIC_CARTPOLE,
            robot=RobotType.CARTPOLE,
            description="test", objectives=[], constraints=[],
        )
        assert _make_task_id(spec) == "RoboSpec-Cartpole-v0"

    def test_anymal_flat_id(self):
        spec = TaskSpec(
            category=TaskCategory.LOCOMOTION_FLAT,
            robot=RobotType.ANYMAL_D,
            description="test", objectives=[], constraints=[],
        )
        assert _make_task_id(spec) == "RoboSpec-Velocity-Flat-Anymal-D-v0"

    def test_anymal_rough_id(self):
        spec = TaskSpec(
            category=TaskCategory.LOCOMOTION_ROUGH,
            robot=RobotType.ANYMAL_D,
            description="test", objectives=[], constraints=[],
        )
        assert _make_task_id(spec) == "RoboSpec-Velocity-Rough-Anymal-D-v0"

    def test_all_ids_start_with_robospec(self):
        """Every task ID must start with 'RoboSpec-' to avoid collisions."""
        for cat in TaskCategory:
            for robot in RobotType:
                spec = TaskSpec(
                    category=cat, robot=robot,
                    description="test", objectives=[], constraints=[],
                )
                task_id = _make_task_id(spec)
                assert task_id.startswith("RoboSpec-"), (
                    f"Task ID '{task_id}' does not start with 'RoboSpec-'"
                )


class TestFindEnvCfgClass:
    def test_finds_class(self):
        code = "class CartpoleCartpoleEnvCfg:\n    pass"
        assert _find_env_cfg_class(code) == "CartpoleCartpoleEnvCfg"

    def test_returns_none_for_no_match(self):
        code = "class MyClass:\n    pass"
        assert _find_env_cfg_class(code) is None

    def test_handles_syntax_error(self):
        code = "class Broken("
        assert _find_env_cfg_class(code) is None


class TestGenerateInitPy:
    def test_contains_gym_register(self):
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_init_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", train_cfg)
        assert 'gym.register(' in result
        assert '"RoboSpec-Cartpole-v0"' in result
        assert 'entry_point="isaaclab.envs:ManagerBasedRLEnv"' in result
        assert "cartpole_cartpole_env_cfg:CartpoleEnvCfg" in result
        assert "disable_env_checker=True" in result

    def test_uses_f_string_name(self):
        train_cfg = CATEGORY_TRAIN_CONFIG["manipulation_reach"]
        result = _generate_init_py("franka_panda_reach", "RoboSpec-Reach-Franka-v0", "FrankaReachEnvCfg", train_cfg)
        assert "{__name__}" in result
        assert "franka_panda_reach_env_cfg:FrankaReachEnvCfg" in result

    def test_contains_agent_config_entry_points(self):
        """__init__.py must include agent config entry points for training."""
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_init_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", train_cfg)
        assert "rl_games_cfg_entry_point" in result
        assert "rsl_rl_cfg_entry_point" in result
        assert "skrl_cfg_entry_point" in result

    def test_reach_agent_configs(self):
        train_cfg = CATEGORY_TRAIN_CONFIG["manipulation_reach"]
        result = _generate_init_py("franka_panda_reach", "RoboSpec-Reach-Franka-v0", "FrankaReachEnvCfg", train_cfg)
        assert "FrankaReachPPORunnerCfg" in result
        assert "skrl_ppo_cfg.yaml" in result

    def test_entry_point_matches_filename(self):
        """The env_cfg_entry_point module must match the actual env_cfg filename."""
        for cat in TaskCategory:
            for robot in RobotType:
                spec = TaskSpec(
                    category=cat, robot=robot,
                    description="test", objectives=[], constraints=[],
                )
                task_name = _make_task_name(spec)
                task_id = _make_task_id(spec)
                train_cfg = CATEGORY_TRAIN_CONFIG.get(cat.value)
                init_py = _generate_init_py(task_name, task_id, "TestEnvCfg", train_cfg)
                expected_module = f"{task_name}_env_cfg"
                assert expected_module in init_py, (
                    f"entry_point module '{expected_module}' not found in __init__.py "
                    f"for {cat.value}/{robot.value}"
                )
                assert expected_module.isidentifier(), (
                    f"Module name '{expected_module}' is not a valid Python identifier"
                )


class TestGenerateTrainPy:
    def test_cartpole_uses_rl_games(self):
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_train_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", 4096, train_cfg)
        assert "rl_games" in result
        assert '"RoboSpec-Cartpole-v0"' in result
        assert '"4096"' in result  # num_envs injected as string arg

    def test_locomotion_uses_rsl_rl(self):
        train_cfg = CATEGORY_TRAIN_CONFIG["locomotion_flat"]
        result = _generate_train_py("anymal_d_flat", "RoboSpec-Velocity-Flat-Anymal-D-v0", "AnymalDFlatEnvCfg", 4096, train_cfg)
        assert "rsl_rl" in result
        assert "AnymalDFlatPPORunnerCfg" in result

    def test_reach_uses_skrl(self):
        train_cfg = CATEGORY_TRAIN_CONFIG["manipulation_reach"]
        result = _generate_train_py("franka_panda_reach", "RoboSpec-Reach-Franka-v0", "FrankaReachEnvCfg", 2048, train_cfg)
        assert "skrl" in result
        assert '"2048"' in result  # num_envs injected as string arg
        assert "FrankaReachPPORunnerCfg" in result

    def test_sys_path_setup(self):
        """train.py must add its directory to sys.path for env config imports."""
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_train_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", 4096, train_cfg)
        assert "sys.path.insert" in result

    def test_gym_register(self):
        """train.py must register the environment with gymnasium."""
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_train_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", 4096, train_cfg)
        assert "gym.register(" in result
        assert "env_cfg_entry_point" in result

    def test_isaaclab_discovery(self):
        """train.py must search for Isaac Lab installation."""
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_train_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", 4096, train_cfg)
        assert "ISAACLAB_DIR" in result
        assert "~/IsaacLab" in result

    def test_no_direct_env_cfg_import(self):
        """train.py must NOT import env_cfg directly (crashes before AppLauncher)."""
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_train_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", 4096, train_cfg)
        assert "from cartpole_cartpole_env_cfg import" not in result
        # But the string entry point must reference it
        assert "cartpole_cartpole_env_cfg:CartpoleEnvCfg" in result

    def test_delegates_to_real_train_script(self):
        """train.py must use runpy to delegate to the real Isaac Lab training script."""
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_train_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", 4096, train_cfg)
        assert "runpy.run_path" in result
        assert "import runpy" in result

    def test_no_applauncher_call(self):
        """train.py must NOT instantiate AppLauncher (the real training script does that)."""
        train_cfg = CATEGORY_TRAIN_CONFIG["classic_cartpole"]
        result = _generate_train_py("cartpole_cartpole", "RoboSpec-Cartpole-v0", "CartpoleEnvCfg", 4096, train_cfg)
        assert "AppLauncher(" not in result
        assert "from isaaclab.app import" not in result

    def test_locomotion_flat_omits_rl_games(self):
        """Locomotion configs have no rl_games agent config — must be omitted."""
        train_cfg = CATEGORY_TRAIN_CONFIG["locomotion_flat"]
        result = _generate_train_py("anymal_d_flat", "RoboSpec-Velocity-Flat-Anymal-D-v0", "AnymalDFlatEnvCfg", 4096, train_cfg)
        assert "rl_games_cfg_entry_point" not in result
        assert "rsl_rl_cfg_entry_point" in result


class TestPostProcessEnvCfg:
    """Tests for the post_process_env_cfg deterministic fixer."""

    def test_replaces_task_specific_mdp_import(self):
        code = (
            "import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp\n"
            "class RewardsCfg:\n"
            "    alive = RewTerm(func=mdp.is_alive)\n"
        )
        result, fixes = post_process_env_cfg(code, "classic_cartpole")
        assert "import isaaclab.envs.mdp as mdp" in result
        assert "isaaclab_tasks.manager_based" not in result
        assert any("mdp import" in f for f in fixes)

    def test_replaces_locomotion_mdp_import(self):
        code = (
            "import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp\n"
            "class RewardsCfg:\n"
            "    track = RewTerm(func=mdp.track_lin_vel_xy_exp)\n"
        )
        result, fixes = post_process_env_cfg(code, "locomotion_flat")
        assert "import isaaclab.envs.mdp as mdp" in result
        assert "isaaclab_tasks" not in result

    def test_injects_robot_import_when_missing(self):
        code = (
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "from isaaclab.utils import configclass\n"
            "\n"
            "@configclass\n"
            "class MySceneCfg:\n"
            "    robot: ArticulationCfg = MISSING\n"
        )
        result, fixes = post_process_env_cfg(code, "classic_cartpole")
        assert "from isaaclab_assets.robots.cartpole import CARTPOLE_CFG" in result
        assert any("CARTPOLE_CFG" in f for f in fixes)

    def test_injects_franka_import(self):
        code = (
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "\n"
            "class MySceneCfg:\n"
            "    pass\n"
        )
        result, fixes = post_process_env_cfg(code, "manipulation_reach")
        assert "FRANKA_PANDA_HIGH_PD_CFG" in result

    def test_injects_anymal_import(self):
        code = (
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "\n"
            "class MySceneCfg:\n"
            "    pass\n"
        )
        result, fixes = post_process_env_cfg(code, "locomotion_rough")
        assert "ANYMAL_D_CFG" in result

    def test_does_not_duplicate_existing_import(self):
        code = (
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip\n"
            "\n"
            "class MySceneCfg:\n"
            "    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path=\"{ENV_REGEX_NS}/Robot\")\n"
        )
        result, fixes = post_process_env_cfg(code, "classic_cartpole")
        assert result.count("CARTPOLE_CFG") == code.count("CARTPOLE_CFG")
        assert not any("Injected" in f for f in fixes)

    def test_replaces_inline_robot_with_replace_pattern(self):
        code = (
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip\n"
            "\n"
            "class MySceneCfg:\n"
            '    robot: ArticulationCfg = AssetBaseCfg(\n'
            '        asset_path="ISAACLAB_NUCLEUS_DIR/Robots/Classic/Cartpole/cartpole.usd",\n'
            "    )\n"
        )
        result, fixes = post_process_env_cfg(code, "classic_cartpole")
        assert "CARTPOLE_CFG.replace(" in result
        assert any("Replaced inline" in f for f in fixes)

    def test_replaces_missing_robot_definition(self):
        code = (
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip\n"
            "\n"
            "class MySceneCfg:\n"
            "    robot: ArticulationCfg = MISSING\n"
        )
        result, fixes = post_process_env_cfg(code, "locomotion_flat")
        assert "ANYMAL_D_CFG.replace(" in result
        assert "MISSING" not in result.split("robot")[1].split("\n")[0]

    def test_warns_on_literal_nucleus_dir_string(self):
        code = (
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            '    terrain = AssetBaseCfg(asset_path="ISAACLAB_NUCLEUS_DIR/Environments/ground.usd")\n'
        )
        _, fixes = post_process_env_cfg(code, "classic_cartpole")
        assert any("ISAACLAB_NUCLEUS_DIR" in f for f in fixes)

    def test_preserves_correct_code(self):
        """Already-correct code should pass through unchanged (except possible import injection)."""
        code = (
            "from isaaclab.envs import ManagerBasedRLEnvCfg\n"
            "from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip\n"
            "import isaaclab.envs.mdp as mdp\n"
            "\n"
            "class MySceneCfg:\n"
            '    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")\n'
        )
        result, fixes = post_process_env_cfg(code, "manipulation_reach")
        assert result == code
        assert fixes == []

    def test_unknown_category_is_noop(self):
        code = "import isaaclab.envs.mdp as mdp\nclass Foo:\n    pass\n"
        result, fixes = post_process_env_cfg(code, "unknown_category")
        assert result == code
        assert fixes == []
