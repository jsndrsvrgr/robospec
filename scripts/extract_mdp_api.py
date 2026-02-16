"""Extract MDP function signatures and docstrings from Isaac Lab source."""

import ast
import os
import textwrap

MDP_DIR = "/tmp/IsaacLab/source/isaaclab/isaaclab/envs/mdp"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "robospec", "knowledge", "api_reference",
)

# Map source files to output markdown files
FILE_MAP = {
    "rewards.py": "mdp_rewards.md",
    "observations.py": "mdp_observations.md",
    "terminations.py": "mdp_terminations.md",
    "events.py": "mdp_events.md",
}

# For actions, we need to scan the actions/ directory
ACTIONS_DIR = os.path.join(MDP_DIR, "actions")


def extract_functions_from_file(filepath: str) -> list[dict]:
    """Extract all top-level function defs with their signatures and docstrings."""
    with open(filepath) as f:
        source = f.read()

    tree = ast.parse(source)
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            # Get the full signature from source lines
            sig = get_signature(node, source)
            docstring = ast.get_docstring(node) or "No description available."
            params = extract_params(node)
            functions.append({
                "name": node.name,
                "signature": sig,
                "docstring": docstring,
                "params": params,
            })

    return functions


def extract_classes_from_file(filepath: str) -> list[dict]:
    """Extract class definitions (for action configs)."""
    with open(filepath) as f:
        source = f.read()

    tree = ast.parse(source)
    classes = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            docstring = ast.get_docstring(node) or "No description available."
            # Get base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.unparse(base))
            classes.append({
                "name": node.name,
                "bases": bases,
                "docstring": docstring,
            })

    return classes


def get_signature(node: ast.FunctionDef, source: str) -> str:
    """Reconstruct function signature from AST."""
    args = []
    all_args = node.args

    # Positional args
    defaults_offset = len(all_args.args) - len(all_args.defaults)
    for i, arg in enumerate(all_args.args):
        annotation = ast.unparse(arg.annotation) if arg.annotation else ""
        name = arg.arg
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(all_args.defaults):
            default = ast.unparse(all_args.defaults[default_idx])
            if annotation:
                args.append(f"{name}: {annotation} = {default}")
            else:
                args.append(f"{name}={default}")
        else:
            if annotation:
                args.append(f"{name}: {annotation}")
            else:
                args.append(name)

    # *args
    if all_args.vararg:
        args.append(f"*{all_args.vararg.arg}")

    # keyword-only args
    kw_defaults = all_args.kw_defaults
    for i, arg in enumerate(all_args.kwonlyargs):
        annotation = ast.unparse(arg.annotation) if arg.annotation else ""
        name = arg.arg
        default = kw_defaults[i]
        if default:
            default_str = ast.unparse(default)
            if annotation:
                args.append(f"{name}: {annotation} = {default_str}")
            else:
                args.append(f"{name}={default_str}")
        else:
            if annotation:
                args.append(f"{name}: {annotation}")
            else:
                args.append(name)

    # **kwargs
    if all_args.kwarg:
        args.append(f"**{all_args.kwarg.arg}")

    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"

    return f"{node.name}({', '.join(args)}){ret}"


def extract_params(node: ast.FunctionDef) -> list[tuple[str, str]]:
    """Extract parameter names and annotations."""
    params = []
    for arg in node.args.args:
        if arg.arg == "self":
            continue
        annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
        params.append((arg.arg, annotation))
    for arg in node.args.kwonlyargs:
        annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
        params.append((arg.arg, annotation))
    return params


def format_function_md(func: dict) -> str:
    """Format a single function as markdown."""
    lines = [f"### mdp.{func['name']}\n"]
    lines.append(f"**Signature:** `{func['signature']}`\n")

    # Clean up docstring - take first paragraph
    doc = func["docstring"]
    first_para = doc.split("\n\n")[0].strip()
    lines.append(f"**Description:** {first_para}\n")

    if func["params"]:
        lines.append("**Parameters:**")
        for name, annotation in func["params"]:
            lines.append(f"- `{name}`: {annotation}")
        lines.append("")

    return "\n".join(lines)


def format_class_md(cls: dict) -> str:
    """Format a class as markdown."""
    lines = [f"### {cls['name']}\n"]
    if cls["bases"]:
        lines.append(f"**Inherits from:** {', '.join(cls['bases'])}\n")
    first_para = cls["docstring"].split("\n\n")[0].strip()
    lines.append(f"**Description:** {first_para}\n")
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract from standard MDP files
    for src_file, out_file in FILE_MAP.items():
        filepath = os.path.join(MDP_DIR, src_file)
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping")
            continue

        functions = extract_functions_from_file(filepath)
        category = src_file.replace(".py", "").replace("_", " ").title()

        md_lines = [f"# Isaac Lab MDP {category}\n"]
        md_lines.append(f"Extracted from `isaaclab.envs.mdp.{src_file.replace('.py', '')}`\n")
        md_lines.append(f"Total functions: {len(functions)}\n")
        md_lines.append("---\n")

        for func in functions:
            md_lines.append(format_function_md(func))
            md_lines.append("---\n")

        out_path = os.path.join(OUTPUT_DIR, out_file)
        with open(out_path, "w") as f:
            f.write("\n".join(md_lines))
        print(f"Wrote {out_path} ({len(functions)} functions)")

    # Extract actions from the actions/ directory
    if os.path.isdir(ACTIONS_DIR):
        all_classes = []
        all_functions = []
        for fname in sorted(os.listdir(ACTIONS_DIR)):
            if fname.endswith(".py") and not fname.startswith("_"):
                fpath = os.path.join(ACTIONS_DIR, fname)
                all_classes.extend(extract_classes_from_file(fpath))
                all_functions.extend(extract_functions_from_file(fpath))

        md_lines = ["# Isaac Lab MDP Actions\n"]
        md_lines.append("Extracted from `isaaclab.envs.mdp.actions`\n")
        md_lines.append(f"Total action classes: {len(all_classes)}\n")
        md_lines.append("---\n")

        for cls in all_classes:
            md_lines.append(format_class_md(cls))
            md_lines.append("---\n")

        if all_functions:
            md_lines.append("\n## Helper Functions\n")
            for func in all_functions:
                md_lines.append(format_function_md(func))
                md_lines.append("---\n")

        out_path = os.path.join(OUTPUT_DIR, "mdp_actions.md")
        with open(out_path, "w") as f:
            f.write("\n".join(md_lines))
        print(f"Wrote {out_path} ({len(all_classes)} classes, {len(all_functions)} functions)")


if __name__ == "__main__":
    main()
