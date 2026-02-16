"""Tests for the task analyzer."""

import json
import pytest

from robospec.pipeline.analyzer import (
    TaskCategory,
    RobotType,
    TaskSpec,
    _extract_json,
    _parse_task_spec,
)


class TestExtractJson:
    """Test JSON extraction from various response formats."""

    def test_clean_json(self):
        raw = '{"category": "manipulation_reach", "robot": "franka_panda"}'
        result = _extract_json(raw)
        assert result["category"] == "manipulation_reach"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"category": "classic_cartpole", "robot": "cartpole"}\n```'
        result = _extract_json(raw)
        assert result["category"] == "classic_cartpole"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the analysis:\n{"category": "locomotion_flat", "robot": "anymal_d"}\nDone.'
        result = _extract_json(raw)
        assert result["category"] == "locomotion_flat"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("This is not JSON at all")


class TestParseTaskSpec:
    """Test TaskSpec parsing from dict."""

    def test_basic_parse(self):
        data = {
            "category": "manipulation_reach",
            "robot": "franka_panda",
            "objectives": ["reach targets"],
            "constraints": ["smooth motion"],
            "difficulty": "easy",
            "episode_length_s": 3.0,
            "num_envs": 2048,
            "custom_notes": None,
        }
        spec = _parse_task_spec(data, "reach targets")
        assert spec.category == TaskCategory.MANIPULATION_REACH
        assert spec.robot == RobotType.FRANKA_PANDA
        assert spec.objectives == ["reach targets"]
        assert spec.difficulty == "easy"
        assert spec.episode_length_s == 3.0

    def test_defaults(self):
        data = {
            "category": "classic_cartpole",
            "robot": "cartpole",
        }
        spec = _parse_task_spec(data, "balance pole")
        assert spec.difficulty == "medium"
        assert spec.episode_length_s == 5.0
        assert spec.num_envs == 4096
        assert spec.objectives == []

    def test_invalid_category_raises(self):
        data = {"category": "flying", "robot": "cartpole"}
        with pytest.raises(ValueError):
            _parse_task_spec(data, "fly")

    def test_invalid_robot_raises(self):
        data = {"category": "classic_cartpole", "robot": "drone"}
        with pytest.raises(ValueError):
            _parse_task_spec(data, "fly")


class TestTaskSpecMapping:
    """Verify expected category-robot mappings (unit tests, no API calls)."""

    @pytest.mark.parametrize(
        "category,robot",
        [
            ("manipulation_reach", "franka_panda"),
            ("classic_cartpole", "cartpole"),
            ("locomotion_flat", "anymal_d"),
            ("locomotion_rough", "anymal_d"),
        ],
    )
    def test_valid_category_robot_pairs(self, category, robot):
        data = {
            "category": category,
            "robot": robot,
            "objectives": ["test"],
            "constraints": [],
        }
        spec = _parse_task_spec(data, "test")
        assert spec.category == TaskCategory(category)
        assert spec.robot == RobotType(robot)
