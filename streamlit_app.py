"""RoboSpec Streamlit Demo UI."""

import asyncio
import io
import zipfile
from pathlib import Path

import streamlit as st

from robospec.nemotron.client import NemotronClient
from robospec.pipeline.analyzer import analyze_task
from robospec.pipeline.context import build_context
from robospec.pipeline.generator import (
    generate_config,
    repair_config,
    _build_config,
    _make_task_name,
    _make_task_id,
)
from robospec.pipeline.validator import (
    validate_config,
    auto_correct_code,
    load_api_whitelist,
)
from robospec.pipeline.explainer import explain_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_REPAIR_ATTEMPTS = 2

EXAMPLE_PROMPTS = [
    "Train a Franka arm to reach random targets smoothly",
    "Balance a pole on a cart",
    "Walk forward on flat ground",
    "Navigate rough terrain with a quadruped",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RoboSpec",
    page_icon="https://raw.githubusercontent.com/nvidia/IsaacLab/main/docs/source/_static/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — shadcn/ui inspired
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp, .stMarkdown, .stText, .stTextInput input, .stTextArea textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* ---------- Page background ---------- */
    .stApp {
        background-color: #fafafa;
    }

    /* ---------- Hide chrome ---------- */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---------- Card containers ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e4e4e7 !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }

    /* ---------- Primary button (dark) ---------- */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #18181b !important;
        color: #fff !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.5rem !important;
        transition: background-color 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #27272a !important;
    }

    /* ---------- Secondary / chip buttons ---------- */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background-color: #f4f4f5 !important;
        color: #3f3f46 !important;
        border: 1px solid #e4e4e7 !important;
        border-radius: 9999px !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        padding: 0.25rem 0.85rem !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #e4e4e7 !important;
        color: #18181b !important;
    }

    /* ---------- Text area ---------- */
    .stTextArea textarea {
        border: 1px solid #e4e4e7 !important;
        border-radius: 0.5rem !important;
        font-size: 14px !important;
        padding: 0.75rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #a1a1aa !important;
        box-shadow: 0 0 0 3px rgba(161,161,170,0.15) !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 1px solid #e4e4e7 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        color: #71717a !important;
    }
    .stTabs [aria-selected="true"] {
        color: #18181b !important;
    }

    /* ---------- Code blocks ---------- */
    .stCodeBlock {
        border-radius: 0.5rem !important;
        border: 1px solid #e4e4e7 !important;
    }

    /* ---------- Download button ---------- */
    .stDownloadButton > button {
        background-color: #18181b !important;
        color: #fff !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #27272a !important;
    }

    /* ---------- Status expander ---------- */
    [data-testid="stStatusWidget"] {
        border: 1px solid #e4e4e7 !important;
        border-radius: 0.75rem !important;
    }

    /* ---------- Divider ---------- */
    hr {
        border: none !important;
        border-top: 1px solid #e4e4e7 !important;
        margin: 1.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def badge(text: str, variant: str = "default") -> str:
    """Return an HTML badge span."""
    colors = {
        "success": ("#16a34a", "#f0fdf4", "#bbf7d0"),
        "error": ("#dc2626", "#fef2f2", "#fecaca"),
        "warning": ("#d97706", "#fffbeb", "#fde68a"),
        "info": ("#2563eb", "#eff6ff", "#bfdbfe"),
        "default": ("#71717a", "#f4f4f5", "#e4e4e7"),
    }
    text_c, bg_c, border_c = colors.get(variant, colors["default"])
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:9999px;'
        f'font-size:12px;font-weight:500;line-height:1.6;'
        f'color:{text_c};background:{bg_c};border:1px solid {border_c};">'
        f'{text}</span>'
    )


def create_zip(config) -> bytes:
    """Build an in-memory zip archive of all generated files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{config.task_name}_env_cfg.py", config.env_cfg)
        if config.init_py:
            zf.writestr("__init__.py", config.init_py)
        if config.train_script:
            zf.writestr("train.py", config.train_script)
        if config.readme:
            zf.writestr("README.md", config.readme)
    return buf.getvalue()


def _build_whitelist_hint(errors: list[str]) -> str:
    """If errors include unknown API symbols, build a hint with the full whitelist."""
    if not any("Unknown MDP function" in e for e in errors):
        return ""
    whitelist = load_api_whitelist()
    return "AVAILABLE MDP FUNCTIONS (use ONLY these):\n" + ", ".join(sorted(whitelist))


# ---------------------------------------------------------------------------
# Init session state
# ---------------------------------------------------------------------------
for key, default in {
    "description": EXAMPLE_PROMPTS[0],
    "generated": False,
    "task_spec": None,
    "config": None,
    "validation": None,
    "corrections": [],
    "repair_log": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(description: str, status_widget):
    """Run the full RoboSpec pipeline, writing progress to status_widget."""
    client = NemotronClient()

    try:
        # 1. Analyze
        status_widget.write("Analyzing task description...")
        task_spec = await analyze_task(client, description)
        st.session_state["task_spec"] = task_spec
        status_widget.write(
            f"Detected **{task_spec.category.value}** with **{task_spec.robot.value}**"
        )

        # 2. Context
        status_widget.write("Building Isaac Lab reference context...")
        context = build_context(task_spec)
        tokens_est = len(context) // 4
        status_widget.write(f"Context ready (~{tokens_est:,} tokens)")

        # 3. Generate
        status_widget.write("Generating Isaac Lab configuration...")
        config = await generate_config(client, task_spec, context)

        # 4. Auto-correct + Validate
        task_name = _make_task_name(task_spec)
        task_id = _make_task_id(task_spec)

        best_code, corrections = auto_correct_code(config.env_cfg)
        st.session_state["corrections"] = corrections
        if corrections:
            status_widget.write(
                f"Auto-corrected {len(corrections)} hallucination(s)"
            )

        status_widget.write("Validating generated code...")
        result = validate_config(best_code)

        # 5. Repair loop (up to 2 attempts)
        repair_log: list[str] = []
        for attempt in range(MAX_REPAIR_ATTEMPTS):
            if result.is_valid:
                break

            msg = f"Repair attempt {attempt + 1}/{MAX_REPAIR_ATTEMPTS}..."
            status_widget.write(msg)
            repair_log.append(msg)

            whitelist_hint = _build_whitelist_hint(result.errors)
            repaired_code = await repair_config(
                client, best_code, result.errors, context, whitelist_hint
            )
            best_code, repair_corrections = auto_correct_code(repaired_code)
            corrections.extend(repair_corrections)

            status_widget.write("Re-validating...")
            result = validate_config(best_code)

        st.session_state["repair_log"] = repair_log
        st.session_state["validation"] = result
        st.session_state["corrections"] = corrections

        # 6. Rebuild config with best code
        config = _build_config(
            best_code,
            config.raw_response,
            task_name,
            task_id,
            task_spec.num_envs,
            task_spec.category.value,
        )

        # 7. Explain
        status_widget.write("Generating reward explanation...")
        config.readme = await explain_config(client, config.env_cfg, description)

        st.session_state["config"] = config
        st.session_state["generated"] = True

    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<div style="padding:1.5rem 0 0.5rem;">'
    '<h1 style="font-size:2rem;font-weight:700;color:#18181b;margin:0;letter-spacing:-0.025em;">'
    'RoboSpec</h1>'
    '<p style="font-size:15px;color:#71717a;margin:4px 0 0;">Describe a robot task. '
    'Get a runnable Isaac Lab RL environment.</p>'
    '<p style="margin-top:8px;">'
    '<span style="display:inline-block;padding:3px 12px;border-radius:9999px;'
    'font-size:11px;font-weight:600;letter-spacing:0.025em;'
    'color:#16a34a;background:#f0fdf4;border:1px solid #bbf7d0;">'
    'NVIDIA GTC 2026 Golden Ticket Submission</span></p>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Demo videos
# ---------------------------------------------------------------------------
demo_col1, demo_col2 = st.columns(2)

with demo_col1:
    with st.container(border=True):
        st.markdown(
            '<p style="font-size:13px;font-weight:600;color:#18181b;margin-bottom:8px;">'
            'Full Demo</p>',
            unsafe_allow_html=True,
        )
        demo_full = Path("assets/demo_full.mp4")
        if demo_full.exists():
            st.video(str(demo_full))
        else:
            st.markdown(
                '<p style="font-size:13px;color:#a1a1aa;text-align:center;padding:2rem 0;">'
                'Video coming soon</p>',
                unsafe_allow_html=True,
            )

with demo_col2:
    with st.container(border=True):
        st.markdown(
            '<p style="font-size:13px;font-weight:600;color:#18181b;margin-bottom:8px;">'
            'CLI Tool</p>',
            unsafe_allow_html=True,
        )
        demo_cli = Path("assets/demo_cli.mp4")
        if demo_cli.exists():
            st.video(str(demo_cli))
        else:
            st.markdown(
                '<p style="font-size:13px;color:#a1a1aa;text-align:center;padding:2rem 0;">'
                'Video coming soon</p>',
                unsafe_allow_html=True,
            )

st.markdown("")  # spacer

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.markdown(
        '<p style="font-size:14px;font-weight:600;color:#18181b;margin-bottom:4px;">'
        'What should the robot learn to do?</p>',
        unsafe_allow_html=True,
    )

    description = st.text_area(
        "task description",
        value=st.session_state["description"],
        height=100,
        label_visibility="collapsed",
        key="description_input",
    )

    # Example chips
    st.markdown(
        '<p style="font-size:12px;color:#a1a1aa;margin:4px 0 6px;">Try an example:</p>',
        unsafe_allow_html=True,
    )
    chip_cols = st.columns(len(EXAMPLE_PROMPTS))
    for i, (col, prompt) in enumerate(zip(chip_cols, EXAMPLE_PROMPTS)):
        with col:
            # Truncate long labels for chip display
            label = prompt if len(prompt) <= 30 else prompt[:27] + "..."
            if st.button(label, key=f"chip_{i}", type="secondary"):
                st.session_state["description"] = prompt
                st.session_state["generated"] = False
                st.rerun()

    st.markdown("")  # spacer

    generate_btn = st.button(
        "Generate Configuration",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

if generate_btn and description.strip():
    st.session_state["description"] = description
    st.session_state["generated"] = False

    with st.status("Generating Isaac Lab configuration...", expanded=True) as status:
        asyncio.run(run_pipeline(description, status))

        if st.session_state.get("validation") and st.session_state["validation"].is_valid:
            status.update(label="Generation complete", state="complete", expanded=False)
        elif st.session_state.get("validation"):
            status.update(
                label="Generation complete (with validation warnings)",
                state="complete",
                expanded=False,
            )
        else:
            status.update(label="Generation failed", state="error", expanded=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.get("generated"):
    task_spec = st.session_state["task_spec"]
    config = st.session_state["config"]
    validation = st.session_state["validation"]
    corrections = st.session_state.get("corrections", [])

    st.markdown("")  # spacer

    # ---- Task Analysis + Validation side by side ----
    col_analysis, col_validation = st.columns(2)

    with col_analysis:
        with st.container(border=True):
            st.markdown(
                '<p style="font-size:13px;font-weight:600;color:#18181b;margin-bottom:8px;">'
                'Task Analysis</p>',
                unsafe_allow_html=True,
            )
            if task_spec:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Category**  \n`{task_spec.category.value}`")
                    st.markdown(f"**Objectives**  \n{', '.join(task_spec.objectives)}")
                with c2:
                    st.markdown(f"**Robot**  \n`{task_spec.robot.value}`")
                    st.markdown(f"**Difficulty**  \n{task_spec.difficulty}")
                if task_spec.constraints:
                    st.markdown(f"**Constraints:** {', '.join(task_spec.constraints)}")

    with col_validation:
        with st.container(border=True):
            st.markdown(
                '<p style="font-size:13px;font-weight:600;color:#18181b;margin-bottom:8px;">'
                'Validation</p>',
                unsafe_allow_html=True,
            )
            if validation:
                if validation.is_valid:
                    st.markdown(badge("Passed", "success"), unsafe_allow_html=True)
                else:
                    st.markdown(badge("Failed", "error"), unsafe_allow_html=True)
                    for e in validation.errors:
                        st.markdown(f"- {e}")
                for w in validation.warnings:
                    st.markdown(
                        badge("Warning", "warning") + f"&nbsp; {w}",
                        unsafe_allow_html=True,
                    )
            if corrections:
                st.markdown(
                    '<p style="font-size:12px;color:#71717a;margin-top:8px;">Auto-corrections:</p>',
                    unsafe_allow_html=True,
                )
                for c in corrections:
                    st.markdown(f"- `{c}`")

    st.markdown("")  # spacer

    # ---- Generated Files ----
    with st.container(border=True):
        st.markdown(
            '<p style="font-size:13px;font-weight:600;color:#18181b;margin-bottom:4px;">'
            'Generated Files</p>',
            unsafe_allow_html=True,
        )

        env_cfg_name = f"{config.task_name}_env_cfg.py"
        tab_cfg, tab_init, tab_train = st.tabs([env_cfg_name, "__init__.py", "train.py"])

        with tab_cfg:
            st.code(config.env_cfg, language="python", line_numbers=True)
        with tab_init:
            st.code(config.init_py or "# Not generated", language="python", line_numbers=True)
        with tab_train:
            st.code(config.train_script or "# Not generated", language="python", line_numbers=True)

    st.markdown("")  # spacer

    # ---- Reward Explanation ----
    if config.readme:
        with st.container(border=True):
            st.markdown(
                '<p style="font-size:13px;font-weight:600;color:#18181b;margin-bottom:8px;">'
                'Reward Explanation</p>',
                unsafe_allow_html=True,
            )
            st.markdown(config.readme)

    st.markdown("")  # spacer

    # ---- Downloads ----
    dl_cols = st.columns([1, 1, 2])
    with dl_cols[0]:
        st.download_button(
            "Download env_cfg.py",
            data=config.env_cfg,
            file_name=f"{config.task_name}_env_cfg.py",
            mime="text/x-python",
            use_container_width=True,
        )
    with dl_cols[1]:
        zip_bytes = create_zip(config)
        st.download_button(
            "Download All (zip)",
            data=zip_bytes,
            file_name=f"{config.task_name}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("")  # spacer

    # ---- Run instructions ----
    with st.container(border=True):
        st.markdown(
            '<p style="font-size:13px;font-weight:600;color:#18181b;margin-bottom:8px;">'
            'How to Train</p>',
            unsafe_allow_html=True,
        )
        st.code(
            f"# Copy the generated files to your Isaac Lab machine, then:\n"
            f"cd <IsaacLab-dir> && ./isaaclab.sh -p path/to/{config.task_name}/train.py --headless",
            language="bash",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-size:12px;color:#a1a1aa;">'
    'Built with NVIDIA Nemotron + Isaac Lab &mdash; '
    'GTC 2026 Golden Ticket Developer Contest</p>',
    unsafe_allow_html=True,
)
