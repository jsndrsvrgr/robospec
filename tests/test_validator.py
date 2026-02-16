"""Tests for the config validator."""

import pytest

from robospec.pipeline.validator import (
    validate_config,
    ValidationResult,
    load_api_whitelist,
    check_api_symbols,
    auto_correct_code,
)


VALID_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs.mdp import rewards as mdp

@configclass
class RewardsCfg:
    reward_1 = RewTerm(func=mdp.is_alive, weight=1.0)

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
'''

SYNTAX_ERROR_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg

class MyEnvCfg(ManagerBasedRLEnvCfg:
    pass
'''

MISSING_ENV_CFG_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg

class MyConfig:
    pass

class RewardsCfg:
    pass

def __post_init__(self):
    pass
'''

MISSING_REWARDS_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        pass
'''

LARGE_WEIGHT_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm

class RewardsCfg:
    big_reward = RewTerm(func=lambda: None, weight=50.0)

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        pass
'''

NO_IMPORTS_CODE = '''
class RewardsCfg:
    pass

class MyEnvCfg:
    def __post_init__(self):
        pass
'''

# Code with only whitelisted MDP functions
WHITELISTED_MDP_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import rewards as mdp
from isaaclab.managers import RewardTermCfg as RewTerm

class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    vel_penalty = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.decimation = 4
'''

# Code with a fake MDP function
FAKE_MDP_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import rewards as mdp
from isaaclab.managers import RewardTermCfg as RewTerm

class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    fake = RewTerm(func=mdp.totally_fake_function, weight=1.0)

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.decimation = 4
'''

# Code with a close-match typo MDP function (action_rate_l3 -> action_rate_l2)
TYPO_MDP_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import rewards as mdp
from isaaclab.managers import RewardTermCfg as RewTerm

class RewardsCfg:
    vel = RewTerm(func=mdp.action_rate_l3, weight=1.0)

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.decimation = 4
'''

# Code with supplemental whitelist functions (should pass)
SUPPLEMENTAL_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import rewards as mdp
from isaaclab.managers import RewardTermCfg as RewTerm

class RewardsCfg:
    reach = RewTerm(func=mdp.position_command_error_tanh, weight=1.0)
    cartpole = RewTerm(func=mdp.joint_pos_target_l2, weight=-1.0)
    air = RewTerm(func=mdp.feet_air_time, weight=0.1)

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.decimation = 4
'''

# Code with command config (should pass with supplemental whitelist)
COMMAND_CONFIG_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import rewards as mdp
from isaaclab.managers import RewardTermCfg as RewTerm

class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(asset_name="robot")

class RewardsCfg:
    track = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.decimation = 4
'''


class TestValidator:
    def test_valid_code(self):
        result = validate_config(VALID_CODE)
        assert result.is_valid is True
        assert result.errors == []

    def test_syntax_error(self):
        result = validate_config(SYNTAX_ERROR_CODE)
        assert result.is_valid is False
        assert any("SyntaxError" in e for e in result.errors)

    def test_missing_env_cfg(self):
        result = validate_config(MISSING_ENV_CFG_CODE)
        assert result.is_valid is False
        assert any("EnvCfg" in e for e in result.errors)

    def test_missing_rewards_cfg(self):
        result = validate_config(MISSING_REWARDS_CODE)
        assert result.is_valid is False
        assert any("RewardsCfg" in e for e in result.errors)

    def test_missing_imports(self):
        result = validate_config(NO_IMPORTS_CODE)
        assert result.is_valid is False
        assert any("import" in e.lower() for e in result.errors)

    def test_large_weight_warning(self):
        result = validate_config(LARGE_WEIGHT_CODE)
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("50.0" in w for w in result.warnings)


class TestApiWhitelist:
    def test_whitelist_loads_nonempty(self):
        whitelist = load_api_whitelist()
        assert len(whitelist) > 0

    def test_whitelist_contains_known_functions(self):
        whitelist = load_api_whitelist()
        assert "is_alive" in whitelist
        assert "lin_vel_z_l2" in whitelist
        assert "is_terminated" in whitelist

    def test_whitelist_contains_supplemental_functions(self):
        """Supplemental whitelist symbols should be in the loaded whitelist."""
        whitelist = load_api_whitelist()
        assert "position_command_error_tanh" in whitelist
        assert "joint_pos_target_l2" in whitelist
        assert "feet_air_time" in whitelist
        assert "UniformVelocityCommandCfg" in whitelist
        assert "UniformPoseCommandCfg" in whitelist

    def test_whitelisted_code_passes(self):
        result = validate_config(WHITELISTED_MDP_CODE)
        api_errors = [e for e in result.errors if "Unknown MDP function" in e]
        assert api_errors == [], f"Unexpected API errors: {api_errors}"

    def test_supplemental_functions_pass(self):
        """Code using supplemental whitelist functions should pass validation."""
        result = validate_config(SUPPLEMENTAL_CODE)
        api_errors = [e for e in result.errors if "Unknown MDP function" in e]
        assert api_errors == [], f"Unexpected API errors: {api_errors}"

    def test_command_config_passes(self):
        """Command configs like UniformVelocityCommandCfg should pass validation."""
        result = validate_config(COMMAND_CONFIG_CODE)
        api_errors = [e for e in result.errors if "Unknown MDP function" in e]
        assert api_errors == [], f"Unexpected API errors: {api_errors}"

    def test_fake_function_fails(self):
        result = validate_config(FAKE_MDP_CODE)
        assert result.is_valid is False
        api_errors = [e for e in result.errors if "Unknown MDP function" in e]
        assert len(api_errors) == 1
        assert "totally_fake_function" in api_errors[0]

    def test_typo_suggests_correction(self):
        """A close match like action_rate_l3 should suggest action_rate_l2."""
        whitelist = load_api_whitelist()
        assert "action_rate_l2" in whitelist, "action_rate_l2 must be in whitelist"
        errors = check_api_symbols(TYPO_MDP_CODE, whitelist)
        assert len(errors) == 1
        assert "Did you mean" in errors[0]
        assert "action_rate_l2" in errors[0]


class TestAutoCorrect:
    def test_corrects_joint_pos_l2(self):
        code = "x = mdp.joint_pos_l2"
        corrected, corrections = auto_correct_code(code)
        assert "mdp.joint_pos_target_l2" in corrected
        assert len(corrections) == 1

    def test_corrects_track_lin_vel_xy(self):
        code = "x = mdp.track_lin_vel_xy"
        corrected, corrections = auto_correct_code(code)
        assert "mdp.track_lin_vel_xy_exp" in corrected
        assert len(corrections) == 1

    def test_corrects_multiple(self):
        code = "x = mdp.joint_pos_l2\ny = mdp.track_ang_vel_z"
        corrected, corrections = auto_correct_code(code)
        assert "mdp.joint_pos_target_l2" in corrected
        assert "mdp.track_ang_vel_z_exp" in corrected
        assert len(corrections) == 2

    def test_no_correction_needed(self):
        code = "x = mdp.is_alive"
        corrected, corrections = auto_correct_code(code)
        assert corrected == code
        assert corrections == []

    def test_does_not_corrupt_valid_names(self):
        """Auto-correct should not touch valid function names that contain correction substrings."""
        code = "x = mdp.track_lin_vel_xy_exp"
        corrected, corrections = auto_correct_code(code)
        assert corrected == code
        assert corrections == []

    def test_corrected_code_passes_whitelist(self):
        """After auto-correction, hallucinated names should pass the whitelist check."""
        code = "x = mdp.joint_pos_l2\ny = mdp.base_lin_vel_z_l2"
        corrected, _ = auto_correct_code(code)
        whitelist = load_api_whitelist()
        errors = check_api_symbols(corrected, whitelist)
        assert errors == [], f"Post-correction errors: {errors}"


class TestCheckApiSymbols:
    def test_empty_code(self):
        errors = check_api_symbols("x = 1", {"is_alive"})
        assert errors == []

    def test_valid_symbol(self):
        code = "x = mdp.is_alive"
        errors = check_api_symbols(code, {"is_alive"})
        assert errors == []

    def test_unknown_symbol(self):
        code = "x = mdp.fake_func"
        errors = check_api_symbols(code, {"is_alive"})
        assert len(errors) == 1
        assert "fake_func" in errors[0]

    def test_deduplicates(self):
        code = "x = mdp.fake_func\ny = mdp.fake_func"
        errors = check_api_symbols(code, {"is_alive"})
        assert len(errors) == 1

    def test_syntax_error_returns_empty(self):
        errors = check_api_symbols("class Broken(", {"is_alive"})
        assert errors == []


# Code with literal ISAACLAB_NUCLEUS_DIR string (should fail)
INLINE_ROBOT_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as mdp

class MySceneCfg:
    robot = ArticulationCfg(
        prim_path="ISAACLAB_NUCLEUS_DIR/Robots/Classic/Cartpole/cartpole.usd",
    )

class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.decimation = 4
'''

# Code with task-specific mdp import (should warn)
TASK_SPECIFIC_MDP_CODE = '''
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.decimation = 4
'''


class TestInlineRobotCheck:
    def test_literal_nucleus_dir_fails_validation(self):
        result = validate_config(INLINE_ROBOT_CODE)
        assert result.is_valid is False
        assert any("ISAACLAB_NUCLEUS_DIR" in e for e in result.errors)

    def test_no_literal_nucleus_in_valid_code(self):
        result = validate_config(VALID_CODE)
        assert not any("ISAACLAB_NUCLEUS_DIR" in e for e in result.errors)


class TestTaskSpecificMdpCheck:
    def test_task_specific_mdp_warns(self):
        result = validate_config(TASK_SPECIFIC_MDP_CODE)
        assert any("Task-specific mdp import" in w for w in result.warnings)

    def test_core_mdp_no_warning(self):
        result = validate_config(VALID_CODE)
        assert not any("Task-specific mdp import" in w for w in result.warnings)
