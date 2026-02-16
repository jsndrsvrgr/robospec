"""Analyze natural language task descriptions into structured TaskSpec."""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from robospec.nemotron.client import NemotronClient


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


# Directory where prompt templates live
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text that may contain surrounding prose."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response:\n{text[:500]}")


# Sensible episode length defaults per category
EPISODE_LENGTH_DEFAULTS: dict[str, float] = {
    "manipulation_reach": 5.0,
    "classic_cartpole": 5.0,
    "locomotion_flat": 20.0,
    "locomotion_rough": 20.0,
}

MAX_EPISODE_LENGTH = 20.0


def _parse_task_spec(data: dict, description: str) -> TaskSpec:
    """Parse a dict into a TaskSpec, handling string enum values."""
    category = TaskCategory(data["category"])

    # Clamp episode length to sensible defaults
    raw_length = data.get("episode_length_s", None)
    default_length = EPISODE_LENGTH_DEFAULTS.get(category.value, 5.0)
    if raw_length is None or raw_length > MAX_EPISODE_LENGTH:
        episode_length_s = default_length
    else:
        episode_length_s = raw_length

    return TaskSpec(
        category=category,
        robot=RobotType(data["robot"]),
        description=description,
        objectives=data.get("objectives", []),
        constraints=data.get("constraints", []),
        difficulty=data.get("difficulty", "medium"),
        episode_length_s=episode_length_s,
        num_envs=data.get("num_envs", 4096),
        custom_notes=data.get("custom_notes"),
    )


async def analyze_task(client: NemotronClient, user_input: str) -> TaskSpec:
    """Analyze a natural language task description and return a TaskSpec.

    Calls Nemotron with the analyze prompt. If JSON parsing fails, retries once
    with an explicit JSON-only instruction.
    """
    system_prompt = (PROMPTS_DIR / "system.txt").read_text()
    analyze_template = (PROMPTS_DIR / "analyze.txt").read_text()
    user_prompt = analyze_template.format(user_input=user_input)

    # First attempt
    response = await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=1024,
    )

    try:
        data = _extract_json(response)
        return _parse_task_spec(data, user_input)
    except (ValueError, KeyError, json.JSONDecodeError):
        pass

    # Retry with stricter instruction
    retry_prompt = (
        "Your previous response was not valid JSON. "
        "Respond with ONLY a valid JSON object. No explanation, no markdown fences, "
        "no text before or after. Just the JSON.\n\n" + user_prompt
    )
    response = await client.generate(
        system_prompt=system_prompt,
        user_prompt=retry_prompt,
        temperature=0.0,
        max_tokens=1024,
    )

    data = _extract_json(response)
    return _parse_task_spec(data, user_input)
