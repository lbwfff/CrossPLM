#!/usr/bin/env python3
"""CrossPLM GUI - Streamlit-based interactive command builder.

Run from the repository root:
    streamlit run gui/app.py

This is an independent external component that does not modify any project code.
It generates CrossPLM CLI commands based on user input.

Files are saved to fixed default locations:
  - Label map YAML  -> Dataset/<name>.yaml
  - Training config -> Outputs/<task>/config.yaml
"""

import streamlit as st
import os
import sys
import re

# PyYAML is required for label-map read/write; requirements.txt lists it but
# runtime may still miss it (e.g. partial pip install). Fail gracefully in the
# browser instead of crashing the whole import.
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

# Must be the first Streamlit call (no st.* before this).
st.set_page_config(page_title="CrossPLM GUI", page_icon="🧬", layout="wide")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# File/directory browser (no extra dependencies)
try:
    from file_picker import file_picker, dir_picker
except ImportError:
    from gui.file_picker import file_picker, dir_picker  # type: ignore


def _to_training_rel(path: str) -> str:
    """Convert a repo-root-relative path for use inside a Training config.

    Training resolves relative `csv_data_path` / `label_map` against the
    Training/ module directory, so a repo-root-relative path like
    "Dataset/mBMRB.csv" must become "../Dataset/mBMRB.csv". Preset names
    (mBMRB/relaxdb/ss3) and absolute paths are returned unchanged.
    """
    p = path.strip()
    if not p or os.path.isabs(p) or p.startswith("../") or p.startswith("./"):
        return p
    if p in ("mBMRB", "relaxdb", "ss3"):
        return p
    return "../" + p


# ---------------------------------------------------------------------------
# 📝 Label Map Module
# ---------------------------------------------------------------------------
# Built-in presets' column names (used when the user picks a preset by name).
PRESET_COLUMNS = {
    "mBMRB": {"sequence_column": "sequence", "label_column": "label"},
    "relaxdb": {"sequence_column": "sequence", "label_column": "label"},
    "ss3": {"sequence_column": "sequence", "label_column": "label"},
}


def _sanitize_filename(name: str) -> str:
    """Sanitize a user-supplied file/task name to prevent traversal and odd chars."""
    name = name.strip()
    # Strip directory components and collapse to a safe stem
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("._")
    return name[:64]


def _format_pipeline_script(cmds: list[str], names: list[str] | None = None) -> str:
    """Format a list of shell commands as a readable script with completion echos.

    Each command is followed by an ``echo "✓ Step i: <name> completed"`` so the
    script is both manually editable and shell-friendly with ``set -e``.
    """
    if not cmds:
        return ""
    if names is None:
        names = [f"step {i+1}" for i in range(len(cmds))]
    # Pad names to cmds length
    if len(names) < len(cmds):
        names = list(names) + [f"step {i+1}" for i in range(len(names), len(cmds))]
    lines: list[str] = ["#!/bin/bash", "set -e", ""]
    for i, (cmd, name) in enumerate(zip(cmds, names), 1):
        # Guard blocks (if [ ! -f ... ]; then ...) may contain newlines — keep as is
        lines.append(cmd)
        lines.append(f'echo "✓ Step {i}: {name} completed"')
        lines.append("")
    lines.append('echo "All steps completed."')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GUI config import/export + query_params persistence
# ---------------------------------------------------------------------------
import json as _json
import time as _time

# Compatibility shim: st.rerun was experimental_rerun in older Streamlit releases.
if not hasattr(st, "rerun"):
    # Streamlit <1.27
    st.rerun = getattr(st, "experimental_rerun", lambda: None)  # type: ignore[attr-defined]


def _safe_rerun():
    """Rerun the Streamlit script with version compatibility."""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            pass

_GUI_CONFIG_KEYS = [
    # Label Map
    "lm_name_input", "lm_seq_col", "lm_lbl_col", "lm_pos_cls", "lm_ignore",
    "lm_num_classes",
    # Training
    "tr_task_name", "tr_task_type", "tr_seed", "tr_label_map", "tr_do_eval",
    # Single – shared
    "si_experiment", "si_label_map", "si_source", "si_layer", "si_lm_mode",
    # Single – per-step toggles (checkboxes)
    # (enumerated dynamically; see _collect_gui_config)
    # Crossing
    "cr_experiment", "cr_source", "cr_method", "cr_cka", "cr_mi", "cr_hm",
    "cr_controls", "cr_norm", "cr_sel_sim", "cr_sel_probe", "cr_sel_class",
    "cr_probe_mode", "cr_class_mode", "cr_lm_mode_a", "cr_lm_mode_b",
    "cr_quick_exp_a", "cr_quick_exp_b",
]


def _collect_gui_config() -> dict:
    """Snapshot session_state into a JSON-serializable dict (include picker selections)."""
    out: dict = {}
    for k, v in list(st.session_state.items()):
        # Skip transient/restore markers and pagination/cwd internals that would bloat or confuse restore
        if k in ("_qp_restored",):
            continue
        if k.startswith("_"):
            continue
        # Picker internals: keep _selected (meaningful user choice), drop cwd/page/filter/bookkeeping
        if k.endswith("_cwd") or k.endswith("_page") or k.endswith("_page_dir") or k.endswith("_show_browser") or k.endswith("_filter") or k.endswith("_filter_dir"):
            continue
        if k.endswith("_manual"):
            # Manual text_input widget state duplicates _selected; skip to avoid conflict
            continue
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            try:
                _json.dumps(v)
                out[k] = v
            except Exception:
                pass
    # class_inputs is already included above via the generic loop; keep explicit for clarity
    return out


def _validate_gui_config(cfg: dict) -> tuple[bool, str]:
    """Lightweight validation before applying an imported config."""
    if not isinstance(cfg, dict):
        return False, "Config must be a JSON object."
    # num_classes must be sane
    if "num_classes" in cfg:
        try:
            nc = int(cfg["num_classes"])
            if not 1 <= nc <= 20:
                return False, f"num_classes out of range: {nc}"
        except Exception:
            return False, "num_classes must be an integer."
    # class_inputs shape check
    if "class_inputs" in cfg:
        ci = cfg["class_inputs"]
        if not isinstance(ci, list) or len(ci) > 20:
            return False, "class_inputs must be a list of at most 20 rows."
        for row in ci:
            if not isinstance(row, dict) or "char" not in row or "class_id" not in row:
                return False, "Each class_inputs row must have 'char' and 'class_id'."
    return True, ""


def _apply_gui_config(cfg: dict):
    """Restore session_state from a previously exported config dict (with validation)."""
    ok, msg = _validate_gui_config(cfg)
    if not ok:
        raise ValueError(msg)
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        # Don't restore non-JSON-serializable widget bookkeeping that would clash
        if k.endswith("_cwd") or k.endswith("_page") or k.endswith("_page_dir") or k.endswith("_manual") or k.endswith("_show_browser") or k.endswith("_filter") or k.endswith("_filter_dir"):
            continue
        st.session_state[k] = v
    # If class_inputs was imported, also ensure num_classes matches its length
    if "class_inputs" in cfg and "num_classes" not in cfg:
        st.session_state["num_classes"] = len(cfg["class_inputs"])


def _maybe_restore_from_query_params():
    """If the URL contains ?config=<base64 json>, restore it once."""
    # Compatibility: st.query_params is >=1.30, older uses st.experimental_get_query_params
    qp = {}
    try:
        qp = st.query_params  # type: ignore[attr-defined]
    except Exception:
        try:
            qp = st.experimental_get_query_params()  # type: ignore[attr-defined]
        except Exception:
            qp = {}
    raw = None
    if isinstance(qp, dict):
        raw = qp.get("config")
        if isinstance(raw, list):
            raw = raw[0] if raw else None
    else:
        try:
            raw = qp.get("config")  # type: ignore[union-attr]
        except Exception:
            raw = None
    if not raw:
        return
    if st.session_state.get("_qp_restored"):
        return
    try:
        import base64

        decoded = base64.urlsafe_b64decode(raw.encode()).decode()
        cfg = _json.loads(decoded)
        _apply_gui_config(cfg)
        st.session_state["_qp_restored"] = True
        try:
            st.toast("GUI config restored from URL.")  # type: ignore[attr-defined]
        except Exception:
            st.success("GUI config restored from URL.")
    except Exception as e:
        st.warning(f"Could not restore config from URL: {e}")


def _set_query_param(key: str, value: str):
    try:
        st.query_params[key] = value  # type: ignore[attr-defined]
        return
    except Exception:
        pass
    try:
        st.experimental_set_query_params(**{key: value})  # type: ignore[attr-defined]
    except Exception:
        pass


def _clear_query_param(key: str):
    try:
        if key in st.query_params:  # type: ignore[attr-defined]
            del st.query_params[key]  # type: ignore[attr-defined]
            return
    except Exception:
        pass
    try:
        st.experimental_set_query_params()  # type: ignore[attr-defined]
    except Exception:
        pass


def _config_to_query_param(cfg: dict) -> str:
    """Encode config dict as a base64 string for query_params."""
    import base64

    raw = _json.dumps(cfg, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _gui_config_sidebar():
    """Render Export/Import GUI config controls in the sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("💾 GUI Config")
    cfg = _collect_gui_config()
    cfg_json = _json.dumps(cfg, indent=2, ensure_ascii=False)

    # Export
    st.sidebar.download_button(
        "📥 Export GUI config (JSON)",
        data=cfg_json,
        file_name=f"crossplm_gui_{_time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
        key="gui_export_cfg",
    )
    # Shareable URL
    if st.sidebar.button("🔗 Copy shareable URL", key="gui_share_url", use_container_width=True):
        try:
            param = _config_to_query_param(cfg)
            st.sidebar.code(f"?config={param[:120]}… (full param copied to query)", language="text")
            _set_query_param("config", param)
            try:
                st.toast("Shareable URL updated — copy the browser address bar.")  # type: ignore[attr-defined]
            except Exception:
                st.sidebar.success("Shareable URL updated — copy the browser address bar.")
        except Exception as e:
            st.sidebar.error(f"Could not encode config: {e}")

    # Import
    uploaded = st.sidebar.file_uploader("📤 Import GUI config (JSON)", type=["json"], key="gui_import_cfg")
    if uploaded is not None:
        try:
            imported = _json.loads(uploaded.getvalue().decode())
            _apply_gui_config(imported)
            st.sidebar.success("Config imported — rerunning…")
            _safe_rerun()
        except Exception as e:
            st.sidebar.error(f"Could not import config: {e}")
    if st.sidebar.button("🧹 Clear GUI state", key="gui_clear_state", use_container_width=True):
        for k in list(st.session_state.keys()):
            if not k.startswith("_"):
                del st.session_state[k]
        try:
            _clear_query_param("config")
        except Exception:
            pass
        _safe_rerun()


def resolve_label_map_info(label_map: str):
    """Read sequence/label columns and an optional csv path from a label map.

    label_map is either a preset name (mBMRB/relaxdb/ss3) or a path to a YAML
    file. Returns (sequence_column, label_column, csv_data_path).
    """
    if not label_map:
        return "sequence", "label", ""
    spec = None
    if label_map in PRESET_COLUMNS:
        col = PRESET_COLUMNS[label_map]
        return col["sequence_column"], col["label_column"], ""
    if label_map.endswith(".yaml") or label_map.endswith(".yml"):
        if os.path.exists(label_map):
            if yaml is None:
                return "sequence", "label", ""
            try:
                with open(label_map) as f:
                    spec = yaml.safe_load(f) or {}
            except Exception:
                spec = None
    if spec:
        return (spec.get("sequence_column", "sequence"),
                spec.get("label_column", "label"),
                spec.get("csv_data_path", ""))
    return "sequence", "label", ""


def _csv_header_columns(csv_path: str) -> list[str]:
    """Read just the header row of a CSV/TSV to get column names (no full read)."""
    if not csv_path or not os.path.exists(csv_path):
        return []
    try:
        import csv as _csv
        with open(csv_path, newline="") as fh:
            first = fh.readline()
        # Separator auto-detect (same heuristic as Single/Training)
        sep = "\t" if first.count("\t") > first.count(",") else ","
        with open(csv_path, newline="") as fh:
            reader = _csv.DictReader(fh, delimiter=sep)
            return list(reader.fieldnames or [])
    except Exception:
        return []


def _validate_csv_columns(csv_path: str, seq_col: str, lbl_col: str) -> tuple[bool, str]:
    """Return (ok, message) after checking header contains required columns."""
    if not csv_path or not os.path.exists(csv_path):
        return True, ""  # not applicable
    cols = _csv_header_columns(csv_path)
    if not cols:
        return True, ""  # unreadable, skip
    missing = []
    if seq_col not in cols:
        missing.append(f"sequence='{seq_col}'")
    if lbl_col not in cols:
        missing.append(f"label='{lbl_col}'")
    if missing:
        return False, f"CSV header missing {', '.join(missing)} — found columns: {cols}"
    return True, ""


def labelmap_module():
    st.header("📝 Label Map Generator")
    st.markdown(
        "Generate a label-map YAML file for your dataset. The label map defines "
        "the CSV columns, the character-to-class mapping, and ignore characters."
    )

    # --- session state init (must happen before widget instantiation) ---
    if "lm_name" not in st.session_state:
        st.session_state.lm_name = ""
    if "num_classes" not in st.session_state:
        st.session_state.num_classes = 2
    if "class_inputs" not in st.session_state:
        st.session_state.class_inputs = [
            {"char": "", "class_id": 0, "class_name": ""},
            {"char": "", "class_id": 1, "class_name": ""},
        ]

    if yaml is None:
        st.error("PyYAML is not installed. Install it with `pip install pyyaml` or `pip install -r gui/requirements.txt`.")
        return

    # --- 1. Dataset Configuration ---
    st.subheader("1. Dataset Configuration")

    raw_name = st.text_input(
        "Dataset Name *", value=st.session_state.lm_name, key="lm_name_input",
        help="Base name of the dataset (e.g. my_dataset -> my_dataset.yaml)"
    )
    st.session_state.lm_name = _sanitize_filename(raw_name)
    if raw_name and st.session_state.lm_name != raw_name:
        st.caption(f"Sanitized to `{st.session_state.lm_name}` (invalid characters/directories are replaced).")

    # Fixed save location
    save_dir = "Dataset/"
    full_path = os.path.join(
        save_dir, f"{st.session_state.lm_name or 'labelmap'}.yaml"
    )
    if save_dir + st.session_state.lm_name + ".yaml" and st.session_state.lm_name and os.path.exists(full_path):
        st.warning(f"⚠️ `{full_path}` already exists — generating will overwrite it. "
                   "Clear or rename if this is not intended.")

    st.caption(f"File will be saved to: `{full_path}`")

    # Validate that the browsed CSV (if any) actually exists on disk
    _full_csv_exists = lambda p: os.path.exists(p) if p.strip() else True  # empty is OK (optional)
    csv_data_path = file_picker(
        "CSV Data Path (optional)", key="lm_csv_path",
        root="Dataset",
        patterns=["*.csv", "*.tsv"],
        initial=f"Dataset/{st.session_state.lm_name}.csv" if st.session_state.lm_name else "",
        help="Browse to select the dataset CSV (e.g. Dataset/my_data.csv). "
             "Stored in the label map so Training/Single can auto-fill it. "
             "You can also type/paste a path directly.",
        placeholder="e.g. Dataset/my_data.csv",
    )
    if csv_data_path.strip() and not _full_csv_exists(csv_data_path):
        st.warning(f"⚠️ Path does not exist: `{csv_data_path}` — the label map will still save it, but training may fail.")

    col1, col2 = st.columns(2)
    with col1:
        sequence_column = st.text_input("Sequence Column", value="sequence",
                                        key="lm_seq_col",
                                        help="CSV column holding the protein sequence")
    with col2:
        label_column = st.text_input("Label Column", value="label", key="lm_lbl_col",
                                     help="CSV column holding the per-residue label string")

    positive_class = st.number_input("Positive Class ID", value=1, key="lm_pos_cls",
                                     help="Class ID of the positive class (e.g. flexible)")

    # --- 2. Class Definitions (single rerun via form + data_editor) ---
    # Per-text-input row previously caused a rerun on every keystroke (focus loss + lag
    # when many rows). Now a form batches small edits and a data_editor keeps the table
    # editable without per-cell reruns; submit applies the changes at once.
    st.subheader("2. Class Definitions")
    st.markdown("Map **label characters → class IDs**. Edit the table below (like a spreadsheet) and click **Apply** — no rerun per keystroke.")

    # Initialize editor state from the scalar num_classes (first load only)
    # We keep two sources in sync: the number input outside the form controls row count,
    # while the data_editor inside the form edits row values without per-cell reruns.
    # Row-count control is decoupled from the data_editor form so the table refreshes
    # immediately when the count changes, without blocking on "Apply mapping".
    # Two independent actions: "resize rows" vs "apply cell edits".
    c_num, c_resize = st.columns([2, 1])
    with c_num:
        num_classes_out = int(st.number_input(
            "Number of Classes", min_value=1, max_value=20,
            value=st.session_state.num_classes, step=1, key="lm_num_classes",
            help="Number of mapping rows. Change the value then click Apply row count — the table below updates immediately.",
        ))
    with c_resize:
        st.write("")  # align button with input label
        st.write("")
        if st.button("↕ Apply row count", key="lm_apply_rows", use_container_width=True,
                     help="Resize the mapping table to the selected number of classes without saving cell edits."):
            delta = num_classes_out - len(st.session_state.class_inputs)
            if delta > 0:
                for _ in range(delta):
                    idx = len(st.session_state.class_inputs)
                    st.session_state.class_inputs.append({"char": "", "class_id": idx, "class_name": ""})
            elif delta < 0:
                st.session_state.class_inputs = st.session_state.class_inputs[:num_classes_out]
            st.session_state.num_classes = num_classes_out
            # Rebuild editor widget so the new row count is visible
            st.session_state.pop("lm_mapping_editor", None)
            _safe_rerun()
    if num_classes_out != len(st.session_state.class_inputs):
        st.caption(f"Table has {len(st.session_state.class_inputs)} rows — click **Apply row count** to resize to {num_classes_out}.")

    import pandas as pd  # type: ignore

    _editor_key = "lm_mapping_editor"

    import pandas as pd

    _editor_key = "lm_mapping_editor"

    def _current_df():
        return pd.DataFrame(
            [{"Character": r["char"], "Class ID": r["class_id"], "Class Name": r["class_name"]}
             for r in st.session_state.class_inputs]
        )

    with st.form("lm_mapping_form", clear_on_submit=False):
        editor_df = st.data_editor(
            _current_df(),
            key=_editor_key,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Character": st.column_config.TextColumn("Character", help="Label character in the CSV (e.g. 'A', '.', '0', '1')", max_chars=4),
                "Class ID": st.column_config.NumberColumn("Class ID", min_value=0, max_value=20, step=1),
                "Class Name": st.column_config.TextColumn("Class Name", help="Optional (e.g. rigid, flexible)"),
            },
            hide_index=False,
        )
        c_apply, c_reset = st.columns([1, 1])
        with c_apply:
            apply = st.form_submit_button("✅ Apply mapping", use_container_width=True)
        with c_reset:
            reset = st.form_submit_button("↩ Reset", use_container_width=True)

    if reset:
        st.session_state.class_inputs = [
            {"char": "", "class_id": i, "class_name": ""} for i in range(st.session_state.num_classes)
        ]
        st.session_state.pop(_editor_key, None)
        _safe_rerun()

    if apply:
        # Commit editor values back to class_inputs (sanitize single-char constraint).
        # Row count is already handled by "Apply row count" above; Apply mapping only saves edits.
        committed = []
        for _, row in editor_df.iterrows():
            ch = str(row["Character"]).strip() if str(row["Character"]) != "nan" else ""
            if ch and len(ch) > 1:
                st.warning(f"Character must be a single character; truncated `{ch}` → `{ch[0]}`.")
                ch = ch[0]
            try:
                cid = int(row["Class ID"])
            except Exception:
                cid = 0
            cn = str(row["Class Name"]).strip() if str(row["Class Name"]) != "nan" else ""
            committed.append({"char": ch, "class_id": cid, "class_name": cn})
        while len(committed) < st.session_state.num_classes:
            committed.append({"char": "", "class_id": len(committed), "class_name": ""})
        st.session_state.class_inputs = committed[: st.session_state.num_classes]
        st.success("Mapping applied — not yet saved; click *Generate Label Map* to write the YAML.")
    # Derive live preview mapping for the Generate section (use editor_df before Apply too, but commit on Apply)
    # Build a preview from the current committed state (so Generate is deterministic)
    class_mapping = {}
    class_names = {}
    for r in st.session_state.class_inputs:
        if r["char"]:
            class_mapping[r["char"]] = r["class_id"]
        if r["class_name"]:
            class_names[str(r["class_id"])] = r["class_name"]
    # Also show a live count so user sees what will be written
    _live_a = len({r["char"] for r in st.session_state.class_inputs if r["char"]})
    st.caption(f"Committed mappings: {len(class_mapping)} characters → {_live_a and 'unique' or '—'}; "
               f"{len([r for r in st.session_state.class_inputs if r['class_name']])} named classes. Edit the table and click **Apply** to update.")

    # --- 3. Ignore Characters ---
    st.subheader("3. Ignore Characters")
    ignore_chars = st.text_input("Ignore Characters", value="_", key="lm_ignore",
                                 help="Characters ignored during training/eval (e.g. '_' or '_.')")

    # --- Generate ---
    st.divider()
    if st.button("⚙️ Generate Label Map", type="primary"):
        if not st.session_state.lm_name.strip():
            st.error("Please fill in the Dataset Name.")
            return
        if not class_mapping:
            st.error("Please fill in at least one Character mapping.")
            return

        # Validate against the project's label-map spec requirements.
        class_ids = sorted(set(class_mapping.values()))
        if class_ids != list(range(len(class_ids))):
            st.error("Class IDs must be contiguous and start at 0 (e.g. 0,1,2). "
                     f"Got {class_ids}. Adjust the Class ID values.")
            return
        if int(positive_class) not in class_ids:
            st.error(f"Positive Class ID {int(positive_class)} is not among the "
                     f"class IDs {class_ids}.")
            return

        # Follow the template field order exactly:
        # sequence_column -> label_column -> positive_class -> class_names
        # -> mapping -> ignore
        labelmap = {
            "sequence_column": sequence_column,
            "label_column": label_column,
            "positive_class": int(positive_class),
            "class_names": class_names,
            "mapping": class_mapping,
            "ignore": ignore_chars,
        }
        # Optional extension field (ignored by the project code, but read by this
        # GUI so the Training module can auto-fill the CSV path).
        if csv_data_path.strip():
            labelmap["csv_data_path"] = csv_data_path.strip()
        yaml_content = yaml.dump(labelmap, default_flow_style=False,
                                 allow_unicode=True, sort_keys=False)

        st.success("Label map generated!")
        st.subheader("Generated YAML")
        st.code(yaml_content, language="yaml")

        # Save to the fixed Dataset/ directory
        filename = f"{st.session_state.lm_name}.yaml"
        full_save = os.path.join(save_dir, filename)
        try:
            os.makedirs(save_dir, exist_ok=True)
            with open(full_save, "w") as f:
                f.write(yaml_content)
            st.success(f"Saved to: `{full_save}`")
        except Exception as e:
            st.error(f"Could not save file: {e}")

        st.download_button("📥 Download YAML", data=yaml_content,
                           file_name=filename, mime="text/yaml")


# ---------------------------------------------------------------------------
# 🏋️ Training Module
# ---------------------------------------------------------------------------
def training_module():
    st.header("🏋️ Training Module")
    st.markdown(
        "Fill in the experiment configuration once to generate the Config file "
        "and the init / train / eval commands."
    )

    # --- session state init (before widget instantiation) ---
    if "tr_csv" not in st.session_state:
        st.session_state.tr_csv = ""
    if "tr_eval_csv" not in st.session_state:
        st.session_state.tr_eval_csv = ""

    # --- 1. Experiment Setup ---
    st.subheader("1. Experiment Setup")
    col1, col2 = st.columns(2)
    with col1:
        raw_task = st.text_input("Task / Experiment Name *", key="tr_task_name",
                                 help="Name of this experiment (creates Outputs/<name>/). "
                                      "Illegal characters are sanitized to '_' — directory traversal (../) is blocked.")
        task_name = _sanitize_filename(raw_task)
        if raw_task and task_name != raw_task:
            st.caption(f"Sanitized to `{task_name}`.")
    with col2:
        task_type = st.selectbox("Task Type", ["token_classification"], key="tr_task_type")
        seed = st.number_input("Random Seed", value=42, key="tr_seed")
    # Model identity (hidden details, rarely changed) — collapsed by default
    with st.expander("Model identity (backbone / save name)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            model_name = st.text_input("Model Name", value="esm2_t6_8M", key="tr_model_name",
                                       help="Name used when saving the model")
        with c2:
            backbone_model_id = st.text_input("Backbone Model ID",
                                              value="facebook/esm2_t6_8M_UR50D",
                                              key="tr_backbone",
                                              help="HuggingFace model ID of the backbone")
    if not task_name:
        task_name = "task"  # placeholder for preview; validated again at Generate
    cfg_preview = f"Outputs/{task_name}/config.yaml"
    cfg_exists = os.path.exists(cfg_preview)
    # Non-selected paths existence hint (helps before Generate)
    if task_name and cfg_exists:
        st.caption(f"⚠️ `{cfg_preview}` already exists — Generate will overwrite it.")
    # Validate preselected / label-map CSVs so missing files are visible before Generate
    _task_csv_exists = st.session_state.tr_csv and os.path.exists(st.session_state.tr_csv) or not st.session_state.tr_csv
    _eval_csv_exists = st.session_state.tr_eval_csv and os.path.exists(st.session_state.tr_eval_csv) or not st.session_state.tr_eval_csv
    # (Hints rendered below near the respective inputs instead of here, to keep context.)

    # --- 2. Data Configuration ---
    st.subheader("2. Data Configuration")

    label_map = st.text_input("Label Map", value="mBMRB", key="tr_label_map",
                              help="Preset name (mBMRB/relaxdb/ss3) or path to a "
                                   "YAML label-map file. The label map provides the "
                                   "sequence/label columns automatically. "
                                   "You can also browse a YAML file below.")

    lm_seq_col, lm_lbl_col, lm_csv = resolve_label_map_info(label_map)

    # Browse a label-map YAML file (alternative to typing the path)
    lm_file = file_picker(
        "Or browse a Label Map YAML", key="tr_label_map_file",
        root="Dataset",
        patterns=["*.yaml", "*.yml"],
        help="Browse Dataset/ for a label-map YAML. If selected, it overrides the text field above.",
        placeholder="e.g. Dataset/my_dataset.yaml",
    )
    if lm_file and lm_file != label_map:
        label_map = lm_file
        lm_seq_col, lm_lbl_col, lm_csv = resolve_label_map_info(label_map)

    # If the label map carries the CSV path, use it directly (no manual input).
    if lm_csv:
        st.session_state.tr_csv = lm_csv
        st.caption(f"Training CSV: `{lm_csv}` *(from label map `{label_map}`)*")
    else:
        st.session_state.tr_csv = file_picker(
            "Training CSV Path *", key="tr_csv_input",
            root="Dataset",
            patterns=["*.csv", "*.tsv"],
            initial=st.session_state.tr_csv or "Dataset/mBMRB.csv",
            help="Browse to select the training CSV. Or add `csv_data_path` to your label-map YAML to auto-fill.",
            placeholder="e.g. Dataset/mBMRB.csv",
        )

    st.caption(f"Columns from label map: `sequence='{lm_seq_col}'`, `label='{lm_lbl_col}'`")

    # --- 3. Training Parameters ---
    st.subheader("3. Training Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        train_ratio = st.slider("Train Ratio", min_value=0.1, max_value=0.95,
                                value=0.9, key="tr_ratio")
        max_seq_length = st.number_input("Max Sequence Length", value=512, key="tr_maxlen")
    with col2:
        train_bs = st.number_input("Train Batch Size", value=8, key="tr_bs")
        train_eval_bs = st.number_input("Val Batch Size", value=8, key="tr_ebs",
                                        help="Batch size for validation during training")
    with col3:
        learning_rate = st.number_input("Learning Rate", value=2e-5, format="%.2e",
                                        key="tr_lr")
        num_epochs = st.number_input("Training Epochs", value=3, key="tr_epochs")

    # Advanced training options (provenance & Phase 0 freeze)
    with st.expander("Advanced training options", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            grad_accum = int(st.number_input("Gradient Accumulation", value=1, min_value=1, key="tr_grad_accum",
                                            help="Accumulate grads over N batches"))
            weight_decay = float(st.number_input("Weight Decay", value=0.01, format="%.3f", key="tr_weight_decay"))
            save_limit = int(st.number_input("Save Total Limit", value=3, min_value=1, key="tr_save_limit",
                                            help="Keep newest N periodic + best-F1 checkpoints"))
            eval_steps = int(st.number_input("Eval Steps", value=500, min_value=1, key="tr_eval_steps",
                                            help="Run evaluation every N steps"))
            save_steps = int(st.number_input("Save Steps", value=1000, min_value=1, key="tr_save_steps",
                                            help="Save periodic checkpoint every N steps"))
        with c2:
            class_weight = st.selectbox("Class Weight", ["inverse", "none", "sqrt", "log"], index=0, key="tr_class_weight",
                                       help="inverse / sqrt / log / none")
            workers = int(st.number_input("DataLoader Workers", value=2, min_value=0, key="tr_workers"))
            # Mixed precision
            fp16 = st.checkbox("FP16 (AMP)", value=False, key="tr_fp16",
                              help="Requires CUDA; auto-disabled on CPU")
            bf16 = st.checkbox("BF16 (AMP Ampere+)", value=False, key="tr_bf16",
                              help="Requires CUDA Ampere+; mutually exclusive with FP16")
            if fp16 and bf16:
                st.warning("⚠️ Enable only one of FP16 / BF16.")
        with c3:
            freeze_backbone = st.checkbox("Freeze Backbone", value=False, key="tr_freeze_backbone",
                                         help="Train only the classifier head (Phase 0 control for pre-existing vs emergent features)")
            freeze_layers = int(st.number_input("Freeze Bottom N Layers", value=0, min_value=0, max_value=6, key="tr_freeze_layers",
                                                help="0 = none; ignored when Freeze Backbone is on"))
            if freeze_backbone and freeze_layers:
                st.caption("`freeze_layers` is ignored when Freeze Backbone is on.")

    # --- 4. Evaluation Options ---
    st.subheader("4. Evaluation Options")
    do_eval = st.checkbox("Generate eval command", value=True, key="tr_do_eval",
                          help="If checked, an evaluation command is generated after training")
    eval_cmd_bs = 8
    if do_eval:
        st.markdown("**Evaluation label map**")
        eval_label_map = st.text_input(
            "Eval Label Map", key="tr_eval_lm",
            value=label_map,
            help="Label map used for evaluation. Defaults to the training label map, "
                 "but you can specify a different one (e.g. a held-out dataset's map)."
        )

        ev_lm_seq_col, ev_lm_lbl_col, ev_lm_csv = resolve_label_map_info(eval_label_map)

        # Browse an eval label-map YAML as alternative
        ev_lm_file = file_picker(
            "Or browse Eval Label Map YAML", key="tr_eval_lm_file",
            root="Dataset",
            patterns=["*.yaml", "*.yml"],
            help="Browse for an eval label-map YAML (overrides the text field above if selected).",
            placeholder="e.g. Dataset/eval_map.yaml",
        )
        if ev_lm_file and ev_lm_file != eval_label_map:
            eval_label_map = ev_lm_file
            ev_lm_seq_col, ev_lm_lbl_col, ev_lm_csv = resolve_label_map_info(eval_label_map)

        # Eval CSV: prefer the eval label map's CSV, else the training CSV, else manual.
        if ev_lm_csv:
            st.session_state.tr_eval_csv = ev_lm_csv
            st.caption(f"Eval CSV: `{ev_lm_csv}` *(from eval label map `{eval_label_map}`)*")
        else:
            st.session_state.tr_eval_csv = file_picker(
                "Eval CSV Path", key="tr_eval_csv_input",
                root="Dataset",
                patterns=["*.csv", "*.tsv"],
                initial=st.session_state.tr_eval_csv or st.session_state.tr_csv or "Dataset/mBMRB.csv",
                help="Browse to select the evaluation CSV. Defaults to the training CSV.",
                placeholder="e.g. Dataset/mBMRB.csv",
            )
        eval_cmd_bs = int(st.number_input("Eval Batch Size", value=8, key="tr_eval_bs",
                                          help="Batch size used during evaluation"))
        st.caption(f"Eval columns: `sequence='{ev_lm_seq_col}'`, `label='{ev_lm_lbl_col}'`.")

    # --- Validation before Generate: show missing-file hints near the inputs ---
    # Training CSV exists? (when csv comes from lm, the caption already excludes the picker;
    # otherwise the picker itself shows "not found". Add explicit warning for the lm-provided path.)
    if 'lm_csv' in locals() and lm_csv and not os.path.exists(lm_csv):
        st.warning(f"⚠️ Label-map CSV path does not exist: `{lm_csv}` — check the YAML.")
    if 'ev_lm_csv' in locals() and ev_lm_csv and not os.path.exists(ev_lm_csv):
        st.warning(f"⚠️ Eval label-map CSV path does not exist: `{ev_lm_csv}`.")

    # Fixed config save location
    st.subheader("5. Config File Location")
    # Use sanitized task_name for preview; it was already sanitized above
    cfg_dir = f"Outputs/{task_name}/" if task_name else "Outputs/"
    cfg_filename = "config.yaml"
    cfg_full = os.path.join(cfg_dir, cfg_filename)
    st.caption(f"Config will be saved to: `{cfg_full}`")
    if task_name and os.path.exists(cfg_full):
        st.warning(f"⚠️ `{cfg_full}` already exists — Generate will overwrite it.")

    # --- Generate ---
    st.divider()
    if st.button("⚙️ Generate Config & Commands", type="primary"):
        if not task_name.strip():
            st.error("Please fill in the Task / Experiment Name.")
            return
        if not st.session_state.tr_csv.strip():
            st.error("Please fill in the Training CSV Path.")
            return

        # Validate mixed precision
        if 'fp16' in locals() and 'bf16' in locals() and fp16 and bf16:
            st.error("Enable only one of FP16 / BF16.")
            return

        config = {
            "task_name": task_name,
            "model_name": model_name,
            "backbone_model_id": backbone_model_id,
            # Training resolves relative paths against its own module dir.
            "csv_data_path": _to_training_rel(st.session_state.tr_csv),
            "sequence_column": lm_seq_col,
            "label_column": lm_lbl_col,
            "train_ratio": train_ratio,
            "task_type": task_type,
            "max_seq_length": int(max_seq_length),
            "mlm_probability": 0.15,
            "per_device_train_batch_size": int(train_bs),
            "per_device_eval_batch_size": int(train_eval_bs),
            "gradient_accumulation_steps": int(grad_accum) if 'grad_accum' in locals() else 1,
            "learning_rate": learning_rate,
            "weight_decay": float(weight_decay) if 'weight_decay' in locals() else 0.01,
            "num_train_epochs": int(num_epochs),
            "max_steps": -1,
            "logging_steps": 10,
            "eval_steps": int(eval_steps) if 'eval_steps' in locals() else 500,
            "save_steps": int(save_steps) if 'save_steps' in locals() else 1000,
            "save_total_limit": int(save_limit) if 'save_limit' in locals() else 3,
            "class_weight_method": str(class_weight) if 'class_weight' in locals() else "inverse",
            "dataloader_num_workers": int(workers) if 'workers' in locals() else 2,
            "fp16": bool(fp16) if 'fp16' in locals() else False,
            "bf16": bool(bf16) if 'bf16' in locals() else False,
            "freeze_backbone": bool(freeze_backbone) if 'freeze_backbone' in locals() else False,
            "freeze_layers": int(freeze_layers) if 'freeze_layers' in locals() else 0,
            "seed": int(seed),
            "label_map": _to_training_rel(label_map),
        }
        config_yaml = yaml.dump(config, default_flow_style=False, allow_unicode=True,
                                sort_keys=False)

        st.success("Configuration generated!")

        st.subheader("📄 Generated Config File")
        st.code(config_yaml, language="yaml")
        try:
            os.makedirs(cfg_dir, exist_ok=True)
            with open(cfg_full, "w") as f:
                f.write(config_yaml)
            st.success(f"Saved to: `{cfg_full}`")
        except Exception as e:
            st.error(f"Could not save config: {e}")

        st.download_button("📥 Download config.yaml", data=config_yaml,
                           file_name=cfg_filename, mime="text/yaml")

        st.subheader("🚀 Generated Commands")

        # `training init` only scaffolds Outputs/<exp>/config.yaml with defaults
        # and would overwrite the file we just wrote above. When the GUI already
        # wrote the desired config, `init` is unnecessary — point the user
        # directly at `train`. Only suggest `init` when the config did not exist
        # before (fresh task) and note the ordering hazard.
        already_exists = os.path.exists(cfg_full)
        init_cmd = f"python crossplm.py training init --task_name {task_name}"
        train_cmd = f"python crossplm.py training train --config {cfg_full}"

        if already_exists:
            st.info(
                f"`{cfg_full}` already existed and was just overwritten with your GUI settings. "
                "Do **not** run `training init` after this — it would reset the config to defaults. "
                "Use `train` directly."
            )
        else:
            st.caption(
                "`training init` only creates `Outputs/<task>/config.yaml` from the template. "
                "The GUI already wrote the file above, so you can skip `init` and go straight to `train` "
                f"(shown for reference: `{init_cmd}`)."
            )

        st.markdown("**Train Model**")
        st.code(train_cmd, language="bash")

        eval_cmd = None
        if do_eval:
            st.markdown("**Evaluate Model**")
            eval_cmd = (f"python crossplm.py training eval \\\n"
                        f"    --checkpoint Outputs/{task_name}/checkpoints/best \\\n"
                        f"    --csv {st.session_state.tr_eval_csv} \\\n"
                        f"    --label_map {eval_label_map} \\\n"
                        f"    --batch_size {eval_cmd_bs}")
            st.code(eval_cmd, language="bash")

        st.subheader("📋 Full Pipeline")
        st.caption("Pipeline script guards `init` so it will not overwrite an existing `config.yaml`. Each step reports completion via echo.")
        _cmds: list[str] = []
        _names: list[str] = []
        _init_block = f"if [ ! -f {cfg_full} ]; then\n  {init_cmd}\nfi"
        _cmds.append(_init_block)
        _names.append("init (guarded)")
        _cmds.append(train_cmd)
        _names.append("train")
        if eval_cmd:
            _clean_eval = eval_cmd.replace(" \\\n    ", " ")
            _cmds.append(_clean_eval)
            _names.append("eval")
        pipeline_script = _format_pipeline_script(_cmds, _names)
        st.code(pipeline_script, language="bash")
        st.download_button(
            label="📥 Download Pipeline Script",
            data=pipeline_script + "\n",
            file_name="run_training.sh",
            mime="text/x-shellscript"
        )


# ---------------------------------------------------------------------------
# 🔬 Single Module
# ---------------------------------------------------------------------------
def _embeddings_dir(experiment: str, source: str, layer: int) -> str:
    """Assemble the embeddings dir from experiment/source/layer (mirrors paths.py)."""
    base = f"Outputs/{experiment}"
    if source:
        base = f"{base}/{source}"
    return f"{base}/embeddings/layer_{layer}"


def single_module():
    st.header("🔬 Single Module (SAE Analysis)")
    st.markdown(
        "SAE interpretability pipeline. Fill in the shared settings once, tick the "
        "steps you want, and adjust each step's own parameters — then generate the "
        "whole set of commands."
    )

    # ------------------------------------------------------------------ #
    # 0. Shared / pipeline-level settings
    # ------------------------------------------------------------------ #
    st.subheader("Pipeline Settings")

    col1, col2 = st.columns(2)
    with col1:
        raw_exp = st.text_input("Experiment", key="si_experiment",
                                help="Experiment name (creates Outputs/<name>/). "
                                     "Shared by every step. Sanitized to prevent traversal.")
        experiment = _sanitize_filename(raw_exp) if raw_exp.strip() else ""
        if raw_exp and experiment != raw_exp:
            st.caption(f"Sanitized to `{experiment}`.")
        raw_lm = st.text_input("Label Map", value="mBMRB", key="si_label_map",
                               help="Preset name (mBMRB/relaxdb/ss3) or a YAML "
                                    "label-map file path. You can also browse a YAML below.")
        label_map = raw_lm
    with col2:
        raw_source = st.text_input("Source", key="si_source",
                                   help="Data-source id (e.g. mbmrb / swissprot). "
                                        "Nests outputs under Outputs/<name>/<source>/.")
        source = _sanitize_filename(raw_source) if raw_source else ""
        if raw_source and source != raw_source:
            st.caption(f"Sanitized to `{source}`.")
        layer = int(st.number_input("Layer", value=6, key="si_layer",
                                    help="Transformer layer to analyze."))

    # Dedicated existence hints for the two most error-prone paths
    if experiment and not os.path.exists(f"Outputs/{experiment}"):
        st.caption(f"_Outputs/{experiment} not yet on disk — will be created at write time._")
    # ckpt / sequences_csv are shared by several steps
    # Browse label-map YAML as alternative to typing the preset — use radio to decouple from text
    lm_mode = st.radio("Label Map source", ["Preset", "Browse YAML"], horizontal=True, key="si_lm_mode")
    if lm_mode == "Browse YAML":
        lm_file = file_picker(
            "Label Map YAML", key="si_label_map_file",
        root="Dataset",
        patterns=["*.yaml", "*.yml"],
        help="Browse Dataset/ for a label-map YAML. If selected, it overrides the text field above "
             "(useful for custom datasets). You can also keep the preset name.",
        placeholder="e.g. Dataset/my_dataset.yaml",
    )
    if lm_mode == "Browse YAML":
        if lm_file and lm_file.endswith((".yaml", ".yml")):
            label_map = lm_file
    # else Preset mode keeps the text input value as-is

    # Browse checkpoint (directory containing the fine-tuned model)
    ckpt_path = dir_picker(
        "Checkpoint Path", key="si_ckpt",
        root="Outputs",
        help="Browse to the fine-tuned checkpoint directory (e.g. Outputs/my_experiment/checkpoints/best). "
             "Used by extract / fidelity / intervention.",
        initial=st.session_state.get("si_ckpt_selected", ""),
    )
    # Also show a compact text fallback is already inside dir_picker (manual input)

    # Browse sequences CSV
    sequences_csv_picked = file_picker(
        "Sequences CSV", key="si_csv",
        root="Dataset",
        patterns=["*.csv", "*.tsv"],
        help="Browse to the dataset CSV (used by extract / analyze features / sequence / coactivation / fidelity / intervention).",
        placeholder="e.g. Dataset/mBMRB.csv",
    )
    # sequences_csv_picked is the source of truth when browsing (always defined)
    sequences_csv = sequences_csv_picked or ""
    # Auto-fill from label-map YAML when nothing browsed yet
    lm_seq_col, lm_lbl_col, lm_csv = resolve_label_map_info(label_map)
    if lm_csv and not sequences_csv:
        sequences_csv = lm_csv
        st.caption(f"Sequences CSV auto-filled from label map: `{lm_csv}`")
    elif sequences_csv:
        st.caption(f"Sequences CSV: `{sequences_csv}` — columns from label map: `sequence='{lm_seq_col}'`, `label='{lm_lbl_col}'`")
        if not os.path.exists(sequences_csv):
            st.warning(f"⚠️ Sequences CSV not found: `{sequences_csv}`")
        else:
            _ok, _msg = _validate_csv_columns(sequences_csv, lm_seq_col, lm_lbl_col)
            if not _ok:
                st.warning(f"⚠️ {_msg}")
            else:
                st.caption("✅ CSV header contains required columns.")
    if ckpt_path and not os.path.exists(ckpt_path):
        st.caption(f"⚠️ Checkpoint not yet on disk: `{ckpt_path}` — will be created by training/extract.")
    # Also validate the header when csv came from label-map auto-fill
    if 'lm_csv' in locals() and lm_csv and os.path.exists(lm_csv):
        if 'lm_seq_col' in locals() and 'lm_lbl_col' in locals():
            _ok2, _msg2 = _validate_csv_columns(lm_csv, lm_seq_col, lm_lbl_col)
            if not _ok2:
                st.warning(f"⚠️ Label-map CSV (auto-filled) header issue: {_msg2}")

    st.divider()

    # ------------------------------------------------------------------ #
    # Step selection
    # ------------------------------------------------------------------ #
    st.subheader("Select Steps")

    # Each entry: (key, label, needs_embeddings_dir, needs_ckpt, needs_csv)
    step_specs = [
        ("extract", "1. Extract Embeddings",
         True, True, True, "Extract per-residue hidden states from the fine-tuned model."),
        ("train_sae", "2. Train SAE",
         True, False, False, "Train a Sparse Autoencoder on the extracted embeddings."),
        ("analyze_features", "3. Analyze Features (task labels)",
         True, False, True, "Align SAE features with the task labels."),
        ("concepts_build", "4a. Build Concept Matrices",
         False, False, False, "Build per-residue concept matrices from a UniProtKB TSV."),
        ("concepts_align", "4b. Align Features to Concepts",
         True, False, False, "Align SAE features to biological concepts."),
        ("concepts_heldout", "4c. Held-out Validation",
         True, False, False, "Unbiased held-out feature-concept validation."),
        ("analyze_sequence", "5. Analyze Sequence (Cohen's d + motif)",
         True, False, True, "Characterize features along the sequence."),
        ("analyze_coactivation", "6. Analyze Co-activation",
         True, False, True, "Compare pairs of features (co-localized vs disjoint)."),
        ("fidelity", "7. Evaluate Fidelity",
         True, True, True, "Validate that the SAE faithfully represents the model."),
        ("intervention", "8. Evaluate Intervention",
         True, True, True, "Causal single-feature steering."),
        ("visualize", "9. Visualize Features",
         True, False, True, "Plot feature activations on sequences."),
    ]

    # Dependency graph: later steps require earlier outputs exist on disk.
    # Visual disable (not just warning) prevents contradictory selections.
    _needs_map = {
        "extract": [],  # standalone (needs checkpoint from Training, but not from Single)
        "train_sae": ["extract"],
        "analyze_features": ["train_sae"],
        "concepts_build": [],
        "concepts_align": ["train_sae"],
        "concepts_heldout": ["concepts_align"],
        "analyze_sequence": ["train_sae"],
        "analyze_coactivation": ["train_sae"],
        "fidelity": ["train_sae"],
        "intervention": ["train_sae"],
        "visualize": ["train_sae"],
    }

    def _deps_met(key: str, sel: dict) -> bool:
        return all(sel.get(d, False) for d in _needs_map.get(key, []))

    selected = {}
    for key, label, need_emb, need_ckpt, need_csv, desc in step_specs:
        deps = _needs_map.get(key, [])
        blocked = any(not selected.get(d, False) for d in deps)
        help_text = desc + (f" — requires: {', '.join(deps)}" if blocked else "")
        selected[key] = st.checkbox(
            f"{label} — {desc}", key=f"si_sel_{key}",
            disabled=blocked, help=help_text if blocked else desc,
        )
        if blocked and selected[key]:
            # Defensive: should not happen due to disabled, but guard if state was pre-set
            selected[key] = False

    st.divider()

    # ------------------------------------------------------------------ #
    # Per-step parameters
    # ------------------------------------------------------------------ #
    params = {}
    common_cmd = []
    if experiment:
        common_cmd += ["--experiment", experiment]
    if source:
        common_cmd += ["--source", source]

    def base_cmd(script):
        return ["python", "crossplm.py", "single", script] + list(common_cmd)

    def show_embeddings(step, key):
        """Auto-derive embeddings dir from pipeline settings; override only if needed."""
        from pathlib import Path as _Path
        default = _embeddings_dir(experiment, source, layer) if experiment else ""
        if not default:
            return None
        # Fallback: if source-nested doesn't exist but flat does, use flat.
        # This handles the common case where task embeddings were extracted flat
        # (no --source) but analysis is run with a source for output separation.
        _default_exists = _Path(default).exists()
        if source and not _default_exists:
            _flat = _embeddings_dir(experiment, "", layer)
            if _flat and _Path(_flat).exists():
                st.caption(f"Embeddings dir (auto-derived): `{_flat}` (source-nested `{default}` not found, using flat)")
                default = _flat
                _default_exists = True
            else:
                st.caption(f"Embeddings dir (auto-derived): `{default}`")
                st.warning(f"⚠️ Embeddings dir not found: `{default}` — check Pipeline Settings or use Override")
        else:
            st.caption(f"Embeddings dir (auto-derived): `{default}`")
            if not _default_exists:
                st.warning(f"⚠️ Embeddings dir not found: `{default}` — will be created by Extract or check Override")
        use_override = st.checkbox("Override embeddings dir", key=f"{key}_override",
                                    help="Check to browse a different embeddings directory than the derived default.")
        if not use_override:
            return default
        picked = dir_picker(
            "Embeddings Dir (override)", key=key,
            root="Outputs",
            help="Browse to a different embeddings directory (e.g. Outputs/<exp>/embeddings/layer_6).",
            initial=default,
        )
        return picked or default

    # --- 1. Extract Embeddings ---
    if selected["extract"]:
        with st.expander("1. Extract Embeddings", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                batch_size = int(st.number_input("Batch Size", value=8, key="si_ext_bs"))
            with c2:
                max_length = int(st.number_input("Max Length", value=512, key="si_ext_ml"))
            with c3:
                n_shards = int(st.number_input("N Shards", value=5, key="si_ext_shards"))
            c1, c2, c3 = st.columns(3)
            with c1:
                min_seq_len = int(st.number_input("Min Seq Len", value=0, key="si_ext_min"))
            with c2:
                max_seq_len = int(st.number_input("Max Seq Len", value=10000, key="si_ext_max"))
            with c3:
                max_seq_opt = st.number_input("Max Sequences (0=none)", value=0, key="si_ext_mseq")
            params["extract"] = base_cmd("extract_embeddings")
            params["extract"] += ["--ckpt_path", ckpt_path] if ckpt_path else []
            params["extract"] += ["--sequences_csv", sequences_csv] if sequences_csv else []
            params["extract"] += ["--layer", str(layer)]
            params["extract"] += ["--label_map", label_map]
            params["extract"] += ["--batch_size", str(batch_size), "--max_length", str(max_length)]
            params["extract"] += ["--n_shards", str(n_shards)]
            params["extract"] += ["--min_seq_len", str(min_seq_len), "--max_seq_len", str(max_seq_len)]
            if max_seq_opt:
                params["extract"] += ["--max_sequences", str(int(max_seq_opt))]

    # --- 2. Train SAE ---
    if selected["train_sae"]:
        with st.expander("2. Train SAE", expanded=True):
            emb = show_embeddings("train_sae", "si_sae_emb")
            c1, c2, c3 = st.columns(3)
            with c1:
                activation_dim = int(st.number_input("Activation Dim", value=320, key="si_sae_adim"))
            with c2:
                dict_size = int(st.number_input("Dict Size", value=640, key="si_sae_dict"))
            with c3:
                steps = int(st.number_input("Steps", value=20000, key="si_sae_steps"))
            c1, c2, c3 = st.columns(3)
            with c1:
                batch_size = int(st.number_input("Batch Size", value=64, key="si_sae_bs"))
            with c2:
                l1_penalty = float(st.number_input("L1 Penalty", value=0.08, format="%.3f",
                                                   key="si_sae_l1"))
            with c3:
                resample = st.number_input("Resample Steps (0=none)", value=2000,
                                           key="si_sae_res")
            recon_loss = st.selectbox("Reconstruction Loss", ["l2", "mse"], key="si_sae_recon")
            params["train_sae"] = base_cmd("train_sae")
            params["train_sae"] += ["--embeddings_dir", emb] if emb else []
            params["train_sae"] += ["--layer", str(layer)]
            params["train_sae"] += ["--activation_dim", str(activation_dim), "--dict_size", str(dict_size)]
            params["train_sae"] += ["--steps", str(steps), "--batch_size", str(batch_size)]
            params["train_sae"] += ["--l1_penalty", str(l1_penalty)]
            if resample:
                params["train_sae"] += ["--resample_steps", str(int(resample))]
            params["train_sae"] += ["--reconstruction_loss", recon_loss]

    # --- 3. Analyze Features ---
    if selected["analyze_features"]:
        with st.expander("3. Analyze Features (task labels)", expanded=True):
            emb = show_embeddings("analyze_features", "si_af_emb")
            c1, c2 = st.columns(2)
            with c1:
                n_top = int(st.number_input("N Top Features", value=50, key="si_af_top"))
            with c2:
                act_thresh = float(st.number_input("Activation Threshold", value=0.05,
                                                   format="%.3f", key="si_af_thr"))
            params["analyze_features"] = base_cmd("analyze_features")
            params["analyze_features"] += ["--embeddings_dir", emb] if emb else []
            params["analyze_features"] += ["--sequences_csv", sequences_csv] if sequences_csv else []
            params["analyze_features"] += ["--label_map", label_map, "--n_top_features", str(n_top)]
            params["analyze_features"] += ["--activation_threshold", str(act_thresh)]

    # Concept build capture for auto extract (source-aware)
    _cb_annotations_tsv = None
    _cb_n_shards = 5
    _cb_max_residues = 510
    _cb_min_seq_len = 0
    _cb_max_seq_len = 10000

    # --- 4a. Concepts build ---
    if selected["concepts_build"]:
        with st.expander("4a. Build Concept Matrices", expanded=True):
            annotations_tsv = file_picker(
                "Annotations TSV *", key="si_cb_tsv",
                root="Dataset",
                patterns=["*.tsv", "*.tsv.gz", "*.csv"],
                help="Browse to the UniProtKB TSV export (e.g. Dataset/uniprotkb_swissprot.tsv or .tsv.gz).",
                placeholder="e.g. Dataset/uniprotkb_swissprot.tsv.gz",
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                n_shards = int(st.number_input("N Shards", value=5, key="si_cb_shards"))
            with c2:
                max_residues = int(st.number_input("Max Residues", value=510, key="si_cb_mres"))
            with c3:
                min_seq_len = int(st.number_input("Min Seq Len", value=0, key="si_cb_min"))
            max_seq_len = int(st.number_input("Max Seq Len", value=10000, key="si_cb_max"))
            # Subcommand must come directly after `analyze_concepts` — argparse treats
            # `--experiment` before the subcommand as an invalid choice for `command`.
            params["concepts_build"] = ["python", "crossplm.py", "single", "analyze_concepts", "build"] + list(common_cmd)
            params["concepts_build"] += ["--annotations_tsv", annotations_tsv] if annotations_tsv else []
            params["concepts_build"] += ["--n_shards", str(n_shards), "--max_residues", str(max_residues)]
            params["concepts_build"] += ["--min_seq_len", str(min_seq_len), "--max_seq_len", str(max_seq_len)]
            # Capture for auto concept-embeddings extraction
            _cb_annotations_tsv = annotations_tsv
            _cb_n_shards = int(n_shards)
            _cb_max_residues = int(max_residues)
            _cb_min_seq_len = int(min_seq_len)
            _cb_max_seq_len = int(max_seq_len)

    # Auto concept-embeddings: when Build is selected, auto generate a concept-specific
    # extract_embeddings that shares the SAME TSV + shard/seq filters as Build.
    # Align / Heldout then point at that dir, not at the task mBMRB embeddings.
    # Policy: source有值时为 f"{source}_concepts"，空值时为 concepts
    _concepts_source = f"{source}_concepts" if source else "concepts"
    _concepts_emb_dir = _embeddings_dir(experiment, _concepts_source, layer) if experiment else ""
    # max_residues is per-protein truncation; extract uses max_length = max_residues + 2 (BOS/EOS)
    _concept_max_length = int(_cb_max_residues) + 2 if isinstance(_cb_max_residues, int) else 512
    if selected["concepts_build"]:
        # Inject an auto step: extract concept embeddings from the SAME annotations_tsv
        # with identical sharding and truncation, Sequence column auto-filled.
        if _cb_annotations_tsv and experiment and ckpt_path:
            auto_extract = base_cmd("extract_embeddings")
            # override experiment/source for concepts, inject TSV + inferred columns
            auto_extract = ["python", "crossplm.py", "single", "extract_embeddings",
                           "--experiment", experiment,
                           "--source", _concepts_source,
                           "--ckpt_path", ckpt_path,
                           "--sequences_csv", _cb_annotations_tsv,
                           "--sequence_column", "Sequence",
                           "--layer", str(layer),
                           "--label_map", label_map,
                           "--max_length", str(_concept_max_length),
                           "--n_shards", str(_cb_n_shards),
                           "--min_seq_len", str(_cb_min_seq_len),
                           "--max_seq_len", str(_cb_max_seq_len)]
            # Keep batch_size consistent with Pipeline Settings default; user can re-run manually
            # with a different batch if needed — not exposed here to avoid extra UI clutter.
            params["extract_concepts"] = auto_extract
        else:
            # Missing ckpt or experiment: still note what will be auto-generated once filled
            pass

    # --- 4b. Concepts align ---
    if selected["concepts_align"]:
        with st.expander("4b. Align Features to Concepts", expanded=True):
            # Concept embeddings are auto-derived from Build — override optional
            st.caption(f"Embeddings (auto-derived from Build): `{_concepts_emb_dir}` — same TSV + shards as `4a Build`")
            _ca_override = st.checkbox("Override concept embeddings dir", key="si_ca_emb_override",
                                        help="Check to browse a different embeddings directory than the auto-derived concept embeddings.")
            if _ca_override:
                _ca_picked = dir_picker("Embeddings Dir (override)", key="si_ca_emb",
                                         root="Outputs",
                                         help="Browse to a different embeddings directory (e.g. Outputs/<exp>/embeddings/layer_6).",
                                         initial=_concepts_emb_dir or "")
                _ca_emb = _ca_picked or _concepts_emb_dir
            else:
                _ca_emb = _concepts_emb_dir
            thresh = st.text_input("Threshold Percents (space-separated)", value="0 0.15 0.5 0.6 0.8",
                                   key="si_ca_thr")
            params["concepts_align"] = ["python", "crossplm.py", "single", "analyze_concepts", "align"] + list(common_cmd)
            # Align consumes the concept-specific embeddings, not the task mBMRB embeddings
            params["concepts_align"] += ["--embeddings_dir", _ca_emb] if _ca_emb else []
            if thresh.strip():
                params["concepts_align"] += ["--threshold_percents"] + thresh.split()
            if not ckpt_path and selected["concepts_build"]:
                st.warning("Auto concept-extract needs a Checkpoint Path (Pipeline Settings) to generate embeddings.")

    # --- 4c. Concepts heldout ---
    if selected["concepts_heldout"]:
        with st.expander("4c. Held-out Validation", expanded=True):
            st.caption(f"Embeddings (auto-derived from Build): `{_concepts_emb_dir}` — same TSV + shards as `4a Build`")
            _ch_override = st.checkbox("Override concept embeddings dir", key="si_ch_emb_override",
                                        help="Check to browse a different embeddings directory than the auto-derived concept embeddings.")
            if _ch_override:
                _ch_picked = dir_picker("Embeddings Dir (override)", key="si_ch_emb",
                                         root="Outputs",
                                         help="Browse to a different embeddings directory.",
                                         initial=_concepts_emb_dir or "")
                _ch_emb = _ch_picked or _concepts_emb_dir
            else:
                _ch_emb = _concepts_emb_dir
            split_mode = st.selectbox("Split Mode", ["half", "alternate"], key="si_ch_split")
            thresh = st.text_input("Threshold Percents (space-separated)", value="0 0.15 0.5 0.6 0.8",
                                   key="si_ch_thr")
            f1_thresh = float(st.number_input("Held-out F1 Threshold", value=0.3, format="%.2f",
                                              key="si_ch_f1"))
            params["concepts_heldout"] = ["python", "crossplm.py", "single", "analyze_concepts", "heldout"] + list(common_cmd)
            params["concepts_heldout"] += ["--embeddings_dir", _ch_emb] if _ch_emb else []
            params["concepts_heldout"] += ["--split_mode", split_mode]
            params["concepts_heldout"] += ["--heldout_f1_threshold", str(f1_thresh)]
            if thresh.strip():
                params["concepts_heldout"] += ["--threshold_percents"] + thresh.split()
            if not ckpt_path and selected["concepts_build"]:
                st.warning("Auto concept-extract needs a Checkpoint Path (Pipeline Settings) to generate embeddings.")

    # --- 5. Analyze Sequence ---
    if selected["analyze_sequence"]:
        with st.expander("5. Analyze Sequence (Cohen's d + motif)", expanded=True):
            emb = show_embeddings("analyze_sequence", "si_as_emb")
            c1, c2 = st.columns(2)
            with c1:
                feat_indices = st.text_input("Feature Indices *", value="42 234", key="si_as_feat",
                                             help="Space-separated, e.g. '375 42' (required)")
            with c2:
                flank = int(st.number_input("Flank", value=5, key="si_as_flank"))
            if not feat_indices.strip():
                st.warning("⚠️ Analyze Sequence needs Feature Indices (e.g. '42 234').")
            else:
                params["analyze_sequence"] = base_cmd("analyze_sequence")
                params["analyze_sequence"] += ["--embeddings_dir", emb] if emb else []
                params["analyze_sequence"] += ["--sequences_csv", sequences_csv] if sequences_csv else []
                params["analyze_sequence"] += ["--label_map", label_map]
                params["analyze_sequence"] += ["--feature_indices"] + feat_indices.split()
                params["analyze_sequence"] += ["--flank", str(flank)]

    # --- 6. Analyze Co-activation ---
    if selected["analyze_coactivation"]:
        with st.expander("6. Analyze Co-activation", expanded=True):
            emb = show_embeddings("analyze_coactivation", "si_ac_emb")
            c1, c2, c3 = st.columns(3)
            with c1:
                feat_a = int(st.number_input("Feature A *", value=0, key="si_ac_a"))
            with c2:
                feat_b = int(st.number_input("Feature B *", value=1, key="si_ac_b"))
            with c3:
                neighborhood = int(st.number_input("Neighborhood", value=5, key="si_ac_nb"))
            params["analyze_coactivation"] = base_cmd("analyze_coactivation")
            params["analyze_coactivation"] += ["--embeddings_dir", emb] if emb else []
            params["analyze_coactivation"] += ["--sequences_csv", sequences_csv] if sequences_csv else []
            params["analyze_coactivation"] += ["--label_map", label_map]
            params["analyze_coactivation"] += ["--feature_a", str(feat_a), "--feature_b", str(feat_b)]
            params["analyze_coactivation"] += ["--neighborhood", str(neighborhood)]

    # --- 7. Evaluate Fidelity ---
    if selected["fidelity"]:
        with st.expander("7. Evaluate Fidelity", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                batch_size = int(st.number_input("Batch Size", value=8, key="si_fid_bs"))
            with c2:
                max_length = int(st.number_input("Max Length", value=512, key="si_fid_ml"))
            with c3:
                max_seq_opt = st.number_input("Max Sequences (0=none)", value=0, key="si_fid_mseq")
            params["fidelity"] = base_cmd("evaluate_fidelity")
            params["fidelity"] += ["--ckpt_path", ckpt_path] if ckpt_path else []
            params["fidelity"] += ["--sequences_csv", sequences_csv] if sequences_csv else []
            params["fidelity"] += ["--layer", str(layer), "--label_map", label_map]
            params["fidelity"] += ["--batch_size", str(batch_size), "--max_length", str(max_length)]
            if max_seq_opt:
                params["fidelity"] += ["--max_sequences", str(int(max_seq_opt))]

    # --- 8. Evaluate Intervention ---
    if selected["intervention"]:
        with st.expander("8. Evaluate Intervention", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                feature_idx = int(st.number_input("Feature Index *", value=0, key="si_int_feat"))
            with c2:
                mode = st.selectbox("Mode", ["zero", "amplify", "set"], key="si_int_mode")
            with c3:
                scale = float(st.number_input("Scale", value=2.0, format="%.2f", key="si_int_scale"))
            c1, c2, c3 = st.columns(3)
            with c1:
                batch_size = int(st.number_input("Batch Size", value=8, key="si_int_bs"))
            with c2:
                max_length = int(st.number_input("Max Length", value=512, key="si_int_ml"))
            with c3:
                max_seq_opt = st.number_input("Max Sequences (0=none)", value=0, key="si_int_mseq")
            params["intervention"] = base_cmd("evaluate_intervention")
            params["intervention"] += ["--ckpt_path", ckpt_path] if ckpt_path else []
            params["intervention"] += ["--sequences_csv", sequences_csv] if sequences_csv else []
            params["intervention"] += ["--feature_idx", str(feature_idx), "--mode", mode, "--scale", str(scale)]
            params["intervention"] += ["--layer", str(layer), "--label_map", label_map]
            params["intervention"] += ["--batch_size", str(batch_size), "--max_length", str(max_length)]
            if max_seq_opt:
                params["intervention"] += ["--max_sequences", str(int(max_seq_opt))]

    # --- 9. Visualize Features ---
    if selected["visualize"]:
        with st.expander("9. Visualize Features", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                feat_indices = st.text_input("Feature Indices (space-separated)", key="si_viz_feat",
                                             help="Empty = auto-select top features.")
            with c2:
                n_features = int(st.number_input("N Features", value=10, key="si_viz_n"))
            c1, c2 = st.columns(2)
            with c1:
                shard = int(st.number_input("Shard", value=0, key="si_viz_shard"))
            with c2:
                filter_seq = st.text_input("Filter Sequence (optional)", key="si_viz_filter_seq",
                                           help="Exact protein sequence to visualize (single sequence, case-insensitive). "
                                                "When given, only that protein is visualized and --shard is auto-corrected. Leave empty for default 3 proteins.",
                                           placeholder="e.g. MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDNPLDAELLA")
            params["visualize"] = base_cmd("visualize_features")
            params["visualize"] += ["--sequences_csv", sequences_csv] if sequences_csv else []
            params["visualize"] += ["--layer", str(layer), "--label_map", label_map]
            params["visualize"] += ["--shard", str(shard), "--n_features", str(n_features)]
            if feat_indices.strip():
                params["visualize"] += ["--feature_indices"] + feat_indices.split()
            if filter_seq.strip():
                # Pass as a single quoted arg (sequence may be long, no spaces)
                params["visualize"] += ["--filter_sequence", filter_seq.strip()]

    # ------------------------------------------------------------------ #
    # Generate commands
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("🚀 Generated Commands")

    ordered_keys = ["extract", "train_sae", "analyze_features", "concepts_build",
                    "extract_concepts",
                    "concepts_align", "concepts_heldout", "analyze_sequence",
                    "analyze_coactivation", "fidelity", "intervention", "visualize"]

    valid = {k: v for k, v in params.items() if v}
    generated = []
    warnings = []
    for key in ordered_keys:
        if key not in valid:
            continue
        cmd = " ".join(valid[key])
        generated.append(cmd)
        st.code(cmd, language="bash")

    # Warn about steps that are missing a shared required value (ckpt / csv / tsv).
    if experiment or any(selected.values()):
        if selected["extract"] and not ckpt_path:
            warnings.append("Extract Embeddings needs a Checkpoint Path.")
        if selected["fidelity"] and not ckpt_path:
            warnings.append("Evaluate Fidelity needs a Checkpoint Path.")
        if selected["intervention"] and not ckpt_path:
            warnings.append("Evaluate Intervention needs a Checkpoint Path.")
        if (selected["extract"] or selected["analyze_features"]
                or selected["analyze_sequence"] or selected["analyze_coactivation"]
                or selected["fidelity"] or selected["intervention"]) and not sequences_csv:
            warnings.append("Selected steps need a Sequences CSV.")
        if selected["concepts_build"] and not st.session_state.get("si_cb_tsv_selected", ""):
            warnings.append("Build Concept Matrices needs an Annotations TSV.")

    for w in warnings:
        st.warning("⚠️ " + w)

    if not generated:
        st.info("Tick at least one step above to generate its command.")

    if generated:
        st.subheader("📋 Full Pipeline")
        _name_map = {
            "extract": "extract_embeddings",
            "train_sae": "train_sae",
            "analyze_features": "analyze_features",
            "concepts_build": "analyze_concepts build",
            "extract_concepts": "extract_concepts (auto)",
            "concepts_align": "analyze_concepts align",
            "concepts_heldout": "analyze_concepts heldout",
            "analyze_sequence": "analyze_sequence",
            "analyze_coactivation": "analyze_coactivation",
            "fidelity": "evaluate_fidelity",
            "intervention": "evaluate_intervention",
            "visualize": "visualize_features",
        }
        _ordered_valid_keys = [k for k in ordered_keys if k in valid]
        _names = [_name_map.get(k, k) for k in _ordered_valid_keys]
        pipeline_script = _format_pipeline_script(generated, _names)
        st.code(pipeline_script, language="bash")
        st.download_button(
            label="📥 Download Pipeline Script",
            data=pipeline_script + "\n",
            file_name="run_single.sh",
            mime="text/x-shellscript"
        )


# ---------------------------------------------------------------------------
# 🔀 Crossing Module
# ---------------------------------------------------------------------------
def _crossing_experiment_options() -> list[str]:
    """List available Outputs/<exp> directories for quick selection."""
    try:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Outputs")
        if not os.path.exists(base):
            base = "Outputs"
        entries = []
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if os.path.isdir(p) and not name.startswith("."):
                entries.append(name)
        return sorted(entries)
    except Exception:
        return []


def _infer_crossing_dirs(exp: str) -> dict[str, str]:
    """Infer canonical sae/embeddings/concepts paths for a given experiment name."""
    if not exp:
        return {}
    exp = _sanitize_filename(exp)
    candidates = {
        "sae": f"Outputs/{exp}/sae",
        "embeddings": f"Outputs/{exp}/embeddings/layer_6",
        "concepts": f"Outputs/{exp}/concepts",
    }
    # Also try source-nested layouts (e.g. Outputs/<exp>/<source>/embeddings/...)
    # by checking if any subdir contains embeddings/concepts; keep the canonical first.
    result = {}
    for k, path in candidates.items():
        if os.path.exists(path):
            result[k] = path
        else:
            # Fallback: search one level of source nesting
            exp_root = f"Outputs/{exp}"
            found = None
            if os.path.exists(exp_root):
                try:
                    for sub in os.listdir(exp_root):
                        alt = os.path.join(exp_root, sub, k if k != "embeddings" else "embeddings/layer_6")
                        if os.path.exists(alt):
                            found = alt
                            break
                except Exception:
                    pass
            if found:
                result[k] = found
    return result


def _crossing_embeddings_picker(label, key, experiment=""):
    """Helper: browse or auto-derive an embeddings/concepts dir for Crossing."""
    return dir_picker(
        label, key=key,
        root="Outputs",
        help="Browse Outputs/ to select the embeddings/concepts directory "
             "(e.g. Outputs/<exp>/embeddings/layer_6 or Outputs/<exp>/concepts).",
        initial="",
    )


def crossing_module():
    st.header("🔀 Crossing Module")
    st.markdown(
        "Cross-model interpretability: **Phase 1** feature alignment (activation + semantic, "
        "controls, heatmap) and **Phase 2** cross-task probes & shared/private feature classification. "
        "Fill in the shared settings once, then tick the analyses you want."
    )

    # Shared settings
    # Quick-select for A/B experiments: populate the four directory pickers
    st.subheader("Shared Settings")
    st.caption("Tip: use the quick-select below to auto-fill SAE/Embeddings/Concepts from existing experiments — or browse manually.")

    _exp_opts_a = ["<browse manually>"] + _crossing_experiment_options()
    _exp_opts_b = ["<browse manually>"] + _crossing_experiment_options()
    cqs, cqa, cqb = st.columns([1, 1, 1])
    with cqs:
        raw_cross = st.text_input("Experiment (output)", key="cr_experiment",
                                  help="Name for Outputs/<name>/crossing/ where results are written. Sanitized to prevent traversal.")
        cross_experiment = _sanitize_filename(raw_cross) if raw_cross else ""
        if raw_cross and cross_experiment != raw_cross:
            st.caption(f"Sanitized to `{cross_experiment}`.")
        if cross_experiment and os.path.exists(f"Outputs/{cross_experiment}/crossing"):
            st.caption(f"⚠️ `Outputs/{cross_experiment}/crossing` already exists — new runs will append/overwrite there.")
    with cqa:
        sel_exp_a = st.selectbox("Experiment A (quick-fill)", options=_exp_opts_a, key="cr_quick_exp_a",
                                 help="Select an existing experiment to auto-fill SAE A / Embeddings A / Concepts A below.")
    with cqb:
        sel_exp_b = st.selectbox("Experiment B (quick-fill)", options=_exp_opts_b, key="cr_quick_exp_b",
                                 help="Select an existing experiment to auto-fill SAE B / Embeddings B / Concepts B below.")

    # Resolve quick-fill inference (does not overwrite an explicit browse selection held in session_state)
    _infer_a = _infer_crossing_dirs(sel_exp_a) if sel_exp_a != "<browse manually>" else {}
    _infer_b = _infer_crossing_dirs(sel_exp_b) if sel_exp_b != "<browse manually>" else {}
    if _infer_a or _infer_b:
        st.caption(
            "Inferred: "
            + ("  •  ".join([
                f"A: `{v}`" for v in _infer_a.values()
            ] + [f"B: `{v}`" for v in _infer_b.values()]) or "_nothing inferred — check that the experiment has sae/embeddings/concepts_")
        )
        if _infer_a:
            for kk, vv in _infer_a.items():
                sk = {"sae": "cr_sae_a_selected", "embeddings": "cr_emb_a_selected", "concepts": "cr_concepts_a_selected"}.get(kk)
                if sk and not st.session_state.get(sk):
                    st.session_state[sk] = vv
        if _infer_b:
            for kk, vv in _infer_b.items():
                sk = {"sae": "cr_sae_b_selected", "embeddings": "cr_emb_b_selected", "concepts": "cr_concepts_b_selected"}.get(kk)
                if sk and not st.session_state.get(sk):
                    st.session_state[sk] = vv

    col1, col2 = st.columns(2)
    with col1:
        # SAE directories (checkpoints) — directories under Outputs
        sae_a = dir_picker("SAE A", key="cr_sae_a", root="Outputs",
                           help="SAE directory for model A (e.g. Outputs/exp_a/sae).", initial="")
        if sae_a and not os.path.exists(sae_a):
            st.warning(f"⚠️ SAE A not found: `{sae_a}`")
        embeddings_a = dir_picker("Embeddings A", key="cr_emb_a", root="Outputs",
                                  help="Embeddings directory for model A (e.g. Outputs/exp_a/.../embeddings/layer_6).", initial="")
        if embeddings_a and not os.path.exists(embeddings_a):
            st.warning(f"⚠️ Embeddings A not found: `{embeddings_a}`")
    with col2:
        raw_source = st.text_input("Source (optional)", key="cr_source",
                                   help="Data-source id suffix if used (rare for Crossing).")
        cross_source = _sanitize_filename(raw_source) if raw_source else ""
        if raw_source and cross_source != raw_source:
            st.caption(f"Sanitized to `{cross_source}`.")
        sae_b = dir_picker("SAE B", key="cr_sae_b", root="Outputs",
                           help="SAE directory for model B (e.g. Outputs/exp_b/sae).", initial="")
        if sae_b and not os.path.exists(sae_b):
            st.warning(f"⚠️ SAE B not found: `{sae_b}`")
        embeddings_b = dir_picker("Embeddings B", key="cr_emb_b", root="Outputs",
                                  help="Embeddings directory for model B.", initial="")
        if embeddings_b and not os.path.exists(embeddings_b):
            st.warning(f"⚠️ Embeddings B not found: `{embeddings_b}`")

    # Label maps + CSVs for Phase 2 (optional for Phase 1)
    st.markdown("**Labels (for Phase 2 probes/classification)** — leave blank to run Phase 1 only.")
    st.markdown("*Label Map source:* use the radio to choose between a preset name and a browsed YAML — this avoids silent override conflicts.")
    c1, c2 = st.columns(2)
    with c1:
        lm_mode_a = st.radio("Label Map A source", ["Preset", "Browse YAML"], horizontal=True, key="cr_lm_mode_a")
        if lm_mode_a == "Preset":
            label_map_a = st.text_input("Label Map A (preset)", value="mBMRB", key="cr_lm_a",
                                        help="Preset name: mBMRB / relaxdb / ss3.")
        else:
            label_map_a = file_picker("Label Map YAML A", key="cr_lm_file_a", root="Dataset",
                                      patterns=["*.yaml", "*.yml"], help="Browse Dataset/ for a YAML label map.", placeholder="e.g. Dataset/my_map.yaml")
            if not label_map_a:
                label_map_a = "mBMRB"
        labels_a = file_picker("Labels CSV A", key="cr_labels_a", root="Dataset", patterns=["*.csv", "*.tsv"],
                               help="Browse to the per-residue label CSV for task A.", placeholder="e.g. Dataset/mBMRB.csv")
    with c2:
        lm_mode_b = st.radio("Label Map B source", ["Preset", "Browse YAML"], horizontal=True, key="cr_lm_mode_b")
        if lm_mode_b == "Preset":
            label_map_b = st.text_input("Label Map B (preset)", value="mBMRB", key="cr_lm_b",
                                        help="Preset name: mBMRB / relaxdb / ss3.")
        else:
            label_map_b = file_picker("Label Map YAML B", key="cr_lm_file_b", root="Dataset",
                                      patterns=["*.yaml", "*.yml"], help="Browse Dataset/ for a YAML label map.", placeholder="e.g. Dataset/my_map.yaml")
            if not label_map_b:
                label_map_b = "mBMRB"
        labels_b = file_picker("Labels CSV B", key="cr_labels_b", root="Dataset", patterns=["*.csv", "*.tsv"],
                               help="Browse to the per-residue label CSV for task B.", placeholder="e.g. Dataset/other.csv")

    # Phase 1 extra: concepts + controls/heatmap toggles
    st.subheader("Phase 1 Options (Feature Similarity)")
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox("Method", ["correlation", "cosine"], key="cr_method")
        use_cka = st.checkbox("Compute CKA", key="cr_cka")
    with c2:
        use_mi = st.checkbox("Compute MI", key="cr_mi")
        with_heatmap = st.checkbox("Save heatmap", value=True, key="cr_hm")
    with c3:
        with_controls = st.checkbox("Run controls (permutation null)", key="cr_controls")
        normalize = st.checkbox("L_inf normalize SAE features", key="cr_norm")

    # Concepts for semantic similarity (optional)
    with st.expander("Semantic similarity (concepts, optional)", expanded=False):
        concepts_a = dir_picker("Concepts A", key="cr_concepts_a", root="Outputs",
                                help="Concepts directory for A (e.g. Outputs/exp_a/concepts). Leave blank to skip semantic.", initial="")
        concepts_b = dir_picker("Concepts B", key="cr_concepts_b", root="Outputs",
                                help="Concepts directory for B. Leave blank to skip semantic.", initial="")
        semantic_mode = st.selectbox("Semantic mode", ["cosine", "jaccard", "pearson"], key="cr_sem_mode")
        combined = st.checkbox("Also compute S_cross = α S_act + β S_sem", key="cr_combined",
                               help="Requires both concepts dirs above.")

    st.divider()
    st.subheader("Select Analyses")
    sel_sim = st.checkbox("1. Feature Similarity (activation + optional semantic/MI/controls/heatmap)", value=True, key="cr_sel_sim")
    sel_probe = st.checkbox("2. Cross-task Probe (A→A / A→B / B→B / B→A + transfer matrix)", key="cr_sel_probe")
    sel_class = st.checkbox("3. Feature Classification (Shared / A-specific / B-specific)", key="cr_sel_class")

    # Per-analysis extra params
    probe_mode = "auto"
    class_mode = "auto"
    if sel_probe:
        with st.expander("Probe options", expanded=True):
            probe_mode = st.selectbox("Mode", ["auto", "aligned", "disjoint"], key="cr_probe_mode",
                                      help="aligned = same proteins (recommended); auto detects via residues.csv.")
            st.caption("Aligned mode requires both models' embeddings to share proteins (overlapping residues.csv) "
                       "and both label CSVs above.")
    if sel_class:
        with st.expander("Classification options", expanded=True):
            class_mode = st.selectbox("Mode", ["auto", "aligned", "cross_sim"], key="cr_class_mode",
                                      help="aligned = per-feature cross-task probes on shared proteins; "
                                           "cross_sim = similarity+F1 for disjoint sets.")
            thresh = float(st.number_input("F1 Threshold (shared)", value=0.3, format="%.2f", key="cr_class_thr"))

    st.divider()
    st.subheader("🚀 Generated Commands")

    common = []
    if cross_experiment:
        common += ["--experiment", cross_experiment]
        if cross_source:
            common += ["--source", cross_source]

    def base_cross(script):
        return ["python", "crossplm.py", "crossing", script] + list(common)

    cmds = []
    warnings = []

    if sel_sim:
        if not sae_a or not sae_b or not embeddings_a or not embeddings_b:
            warnings.append("Feature Similarity needs SAE A/B and Embeddings A/B.")
        else:
            cmd = base_cross("compute_feature_similarity")
            cmd += ["--sae_a", sae_a, "--sae_b", sae_b]
            cmd += ["--embeddings_a", embeddings_a, "--embeddings_b", embeddings_b]
            cmd += ["--method", method]
            if use_cka:
                cmd += ["--use_cka"]
            if use_mi:
                cmd += ["--use_mi"]
            if normalize:
                cmd += ["--normalize"]
            if with_controls:
                cmd += ["--with_controls"]
            if with_heatmap:
                cmd += ["--with_heatmap"]
            if concepts_a and concepts_b:
                cmd += ["--concepts_a", concepts_a, "--concepts_b", concepts_b,
                        "--semantic_mode", semantic_mode]
                if combined:
                    cmd += ["--combined"]
            cmds.append(" ".join(cmd))
            st.code(" ".join(cmd), language="bash")

    if sel_probe:
        if not sae_a or not sae_b or not embeddings_a or not embeddings_b:
            warnings.append("Cross-task Probe needs SAE A/B and Embeddings A/B (plus labels for aligned mode).")
        else:
            cmd = base_cross("cross_task_probe")
            cmd += ["--sae_a", sae_a, "--sae_b", sae_b]
            cmd += ["--embeddings_a", embeddings_a, "--embeddings_b", embeddings_b]
            if labels_a:
                cmd += ["--labels_a", labels_a]
            if labels_b:
                cmd += ["--labels_b", labels_b]
            if label_map_a:
                cmd += ["--label_map_a", label_map_a]
            if label_map_b:
                cmd += ["--label_map_b", label_map_b]
            cmd += ["--mode", probe_mode]
            cmds.append(" ".join(cmd))
            st.code(" ".join(cmd), language="bash")

    if sel_class:
        if not sae_a or not sae_b or not embeddings_a or not embeddings_b or not labels_a or not labels_b:
            warnings.append("Feature Classification needs SAE A/B, Embeddings A/B, and Labels A/B.")
        else:
            cmd = base_cross("classify_features")
            cmd += ["--sae_a", sae_a, "--sae_b", sae_b]
            cmd += ["--embeddings_a", embeddings_a, "--embeddings_b", embeddings_b]
            cmd += ["--labels_a", labels_a, "--labels_b", labels_b]
            if label_map_a:
                cmd += ["--label_map_a", label_map_a]
            if label_map_b:
                cmd += ["--label_map_b", label_map_b]
            cmd += ["--mode", class_mode]
            if 'thresh' in locals():
                cmd += ["--threshold", str(thresh)]
            cmds.append(" ".join(cmd))
            st.code(" ".join(cmd), language="bash")

    for w in warnings:
        st.warning("⚠️ " + w)

    if not cmds:
        st.info("Tick at least one analysis above to generate its command.")

    if cmds:
        st.subheader("📋 Full Pipeline")
        # Map cmds order to readable names for echo
        _cr_names: list[str] = []
        if sel_sim:
            _cr_names.append("compute_feature_similarity")
        if sel_probe:
            _cr_names.append("cross_task_probe")
        if sel_class:
            _cr_names.append("classify_features")
        pipeline_script = _format_pipeline_script(cmds, _cr_names)
        st.code(pipeline_script, language="bash")
        st.download_button(
            label="📥 Download Pipeline Script",
            data=pipeline_script + "\n",
            file_name="run_crossing.sh",
            mime="text/x-shellscript"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _maybe_restore_from_query_params()
    st.title("🧬 CrossPLM Interactive Command Builder")
    st.markdown("Build CrossPLM commands without typing verbose parameters.")

    st.sidebar.header("Modules")
    module = st.sidebar.radio(
        "Select Module",
        ["📝 Label Map", "🏋️ Training", "🔬 Single", "🔀 Crossing"],
    )

    st.sidebar.divider()
    st.sidebar.markdown("""
    **Quick Start:**
    1. Generate a Label Map first
    2. Use Training to fine-tune a PLM
    3. Use Single for SAE analysis
    """)

    _gui_config_sidebar()

    if module == "📝 Label Map":
        labelmap_module()
    elif module == "🏋️ Training":
        training_module()
    elif module == "🔬 Single":
        single_module()
    elif module == "🔀 Crossing":
        crossing_module()


if __name__ == "__main__":
    main()