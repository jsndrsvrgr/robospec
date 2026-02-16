"""RoboSpec CLI — Natural Language to Isaac Lab Environments."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from robospec.nemotron.client import NemotronClient
from robospec.pipeline.analyzer import analyze_task, RobotType
from robospec.pipeline.context import build_context
from robospec.pipeline.generator import generate_config, repair_config, _build_config, _make_task_name, _make_task_id
from robospec.pipeline.validator import validate_config, load_api_whitelist, auto_correct_code
from robospec.pipeline.explainer import explain_config

MAX_REPAIR_ATTEMPTS = 2

app = typer.Typer(
    name="robospec",
    help="Natural language to Isaac Lab RL environments.",
    add_completion=False,
)
console = Console()


@app.callback()
def callback() -> None:
    """RoboSpec — Natural Language to Isaac Lab Environments."""


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


def _build_whitelist_hint(errors: list[str]) -> str:
    """If errors include unknown API symbols, build a hint with the full whitelist."""
    has_api_errors = any("Unknown MDP function" in e for e in errors)
    if not has_api_errors:
        return ""
    whitelist = load_api_whitelist()
    return "AVAILABLE MDP FUNCTIONS (use ONLY these):\n" + ", ".join(sorted(whitelist))


async def _run_pipeline(
    description: str,
    output: Path,
    robot: Optional[str],
    verbose: bool,
) -> None:
    client = NemotronClient()

    try:
        # 1. Analyze
        with _make_progress() as progress:
            progress.add_task("Analyzing task...", total=None)
            task_spec = await analyze_task(client, description)

        # Override robot if specified
        if robot:
            task_spec.robot = RobotType(robot)

        console.print(
            f"  Detected: [bold cyan]{task_spec.category.value}[/] "
            f"with [bold green]{task_spec.robot.value}[/]"
        )
        console.print(f"  Objectives: {', '.join(task_spec.objectives)}")
        if task_spec.constraints:
            console.print(f"  Constraints: {', '.join(task_spec.constraints)}")

        # 2. Build context
        with _make_progress() as progress:
            progress.add_task("Building context from Isaac Lab reference...", total=None)
            context = build_context(task_spec)

        context_tokens = len(context) // 4  # rough estimate
        console.print(f"  Context: ~{context_tokens:,} tokens of Isaac Lab reference")

        # 3. Generate
        with _make_progress() as progress:
            progress.add_task("Generating Isaac Lab configuration...", total=None)
            config = await generate_config(client, task_spec, context)

        if verbose:
            console.print("\n[dim]--- Raw Nemotron Response ---[/dim]")
            console.print(config.raw_response[:2000])
            console.print("[dim]--- End Response ---[/dim]\n")

        # 4. Auto-correct + Validate + Repair loop
        task_name = _make_task_name(task_spec)
        task_id = _make_task_id(task_spec)

        # Auto-correct common hallucinations before validation
        best_code, corrections = auto_correct_code(config.env_cfg)
        if corrections:
            for c in corrections:
                console.print(f"  [cyan]Auto-corrected: {c}[/]")

        with _make_progress() as progress:
            progress.add_task("Validating generated code...", total=None)
            result = validate_config(best_code)

        if result.warnings:
            for w in result.warnings:
                console.print(f"  [yellow]Warning: {w}[/]")
        for attempt in range(MAX_REPAIR_ATTEMPTS):
            if result.is_valid:
                break

            for e in result.errors:
                console.print(f"  [red]Error: {e}[/]")
            console.print(f"  [yellow]Repair attempt {attempt + 1}/{MAX_REPAIR_ATTEMPTS}...[/]")

            whitelist_hint = _build_whitelist_hint(result.errors)

            with _make_progress() as progress:
                progress.add_task("Repairing configuration...", total=None)
                repaired_code = await repair_config(
                    client, best_code, result.errors, context, whitelist_hint
                )

            best_code, repair_corrections = auto_correct_code(repaired_code)
            if repair_corrections:
                for c in repair_corrections:
                    console.print(f"  [cyan]Auto-corrected: {c}[/]")

            with _make_progress() as progress:
                progress.add_task("Re-validating...", total=None)
                result = validate_config(best_code)

        if not result.is_valid:
            for e in result.errors:
                console.print(f"  [red]Error (final): {e}[/]")
            console.print(
                "[red]Validation still failing after repairs. Saving best attempt for inspection.[/]"
            )

        # Rebuild config with the (possibly repaired) env_cfg
        config = _build_config(
            best_code, config.raw_response,
            task_name, task_id, task_spec.num_envs, task_spec.category.value,
        )

        # 5. Explain
        with _make_progress() as progress:
            progress.add_task("Generating reward explanation...", total=None)
            config.readme = await explain_config(client, config.env_cfg, description)

        # 6. Write output
        out_dir = output or Path("output") / config.task_name
        out_dir.mkdir(parents=True, exist_ok=True)

        files_written: list[str] = []

        env_cfg_name = f"{config.task_name}_env_cfg.py"
        (out_dir / env_cfg_name).write_text(config.env_cfg)
        files_written.append(env_cfg_name)

        if config.init_py:
            (out_dir / "__init__.py").write_text(config.init_py)
            files_written.append("__init__.py")

        if config.train_script:
            (out_dir / "train.py").write_text(config.train_script)
            files_written.append("train.py")

        (out_dir / "README.md").write_text(config.readme)
        files_written.append("README.md")

        # 7. Summary
        console.print()
        file_list = "\n".join(f"   - {f}" for f in files_written)
        console.print(
            Panel(
                f"[bold green]Generated {len(files_written)} files in {out_dir}/[/]\n"
                f"{file_list}\n\n"
                f"To train: scp this folder to an Isaac Lab machine and run:\n"
                f"   [bold]cd <IsaacLab-dir> && ./isaaclab.sh -p {out_dir}/train.py --headless[/]",
                title="[bold]RoboSpec[/]",
                border_style="green",
            )
        )

    finally:
        await client.close()


@app.command()
def generate(
    description: str = typer.Argument(
        ..., help="Natural language description of the robot learning task"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory (default: ./output/{task_name}/)"
    ),
    robot: Optional[str] = typer.Option(
        None,
        "--robot",
        "-r",
        help="Override robot selection (franka_panda, cartpole, anymal_d)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print full Nemotron responses"
    ),
) -> None:
    """Generate an Isaac Lab RL environment from a natural language description."""
    console.print(
        Panel(
            "[bold]RoboSpec[/] — Natural Language to Isaac Lab Environments",
            border_style="blue",
        )
    )
    console.print()

    asyncio.run(_run_pipeline(description, output, robot, verbose))


if __name__ == "__main__":
    app()
