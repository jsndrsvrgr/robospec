"""Generate human-readable explanations of reward design."""

from pathlib import Path

from robospec.nemotron.client import NemotronClient

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def explain_config(
    client: NemotronClient,
    env_cfg: str,
    description: str,
) -> str:
    """Generate a markdown explanation of the reward design.

    Args:
        client: Nemotron API client.
        env_cfg: The generated environment config Python code.
        description: The original user task description.

    Returns:
        Markdown string explaining the reward design.
    """
    system_prompt = (PROMPTS_DIR / "system.txt").read_text()
    explain_template = (PROMPTS_DIR / "explain.txt").read_text()

    # Use replace instead of .format() because env_cfg contains braces
    user_prompt = explain_template.replace("{description}", description).replace(
        "{env_cfg}", env_cfg
    )

    return await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=4096,
    )
