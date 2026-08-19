"""Reusable file/directory picker for Streamlit (no extra dependencies).

Provides two widgets:
- file_picker(label, root, patterns, key): browse and select a file
- dir_picker(label, root, key): browse and select a directory

Both render a collapsible directory browser with navigation. The selected
path is returned as a repo-relative string (or "" if nothing selected).
State is held in st.session_state[key].

Features:
- Root-constrained browsing (cannot escape above root)
- Pattern filtering for files
- Breadcrumb navigation, parent directory button
- Manual path fallback via text_input
- Handles empty/missing directories gracefully
- Default-expanded browser that auto-collapses after selection
- Search filter + pagination for large directories (no endless long menu)
- Sanitized widget keys, path-existence hints, st.container compatibility
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st


# How many entries to show per page before pagination kicks in.
_PAGE_SIZE = 40

# Max columns for breadcrumb before switching to compact mode.
_MAX_BREADCRUMB_COLS = 6


def _repo_root() -> Path:
    """Repository root (parent of gui/)."""
    # Preferred: parent of gui/ (works for `streamlit run gui/app.py`).
    # Fallback: cwd (works for `python -m gui.app` / zipped).
    candidate = Path(__file__).resolve().parent.parent
    if candidate.exists():
        return candidate
    return Path.cwd()


def _normalize_root(root: Optional[str]) -> Path:
    if not root:
        return _repo_root()
    p = Path(root)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve()


def _to_repo_rel(path: Path) -> str:
    """Convert absolute path to repo-relative string for CLI usage."""
    repo = _repo_root().resolve()
    try:
        rel = path.resolve().relative_to(repo)
        return str(rel)
    except ValueError:
        # Outside repo (e.g. /tmp) – return absolute
        return str(path.resolve())


def _sanitize_key_segment(s: str) -> str:
    """Make a string safe for use inside a Streamlit widget key."""
    # Keep alnum and _.-, replace the rest with _
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:48]


def _entry_key(prefix: str, entry: Path, cwd: Path) -> str:
    """Unique, sanitized key for a directory entry (avoids duplicate-widget errors)."""
    rel = _to_repo_rel(entry)
    h = hashlib.sha1(rel.encode()).hexdigest()[:8]
    return f"{prefix}_{_sanitize_key_segment(cwd.name)}_{_sanitize_key_segment(entry.name)}_{h}"


def _path_exists(path_str: str) -> bool:
    """Whether a repo-relative or absolute path currently exists on disk."""
    if not path_str:
        return False
    p = Path(path_str)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.exists()


def _bordered_container():
    """Compatibility shim for `st.container(border=True)` (<1.29 may not support `border`)."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


# Cache directory listings so each rerun doesn't re-scan the same directory.
# TTL keeps things fresh if files are added while the GUI is open.
@st.cache_data(ttl=30, show_spinner=False)
def _cached_list_dir(path_str: str, include_dirs: bool, include_files: bool, patterns_key: str):
    """Cached wrapper around _list_dir_inner (path_str must be absolute string)."""
    path = Path(path_str)
    patterns = patterns_key.split("|") if patterns_key else None
    if patterns == [""]:
        patterns = None
    return _list_dir_inner(path, include_dirs, include_files, patterns)


def _list_dir_inner(path: Path, include_dirs: bool, include_files: bool, patterns: Optional[List[str]] = None):
    """List directory entries, partitioned into dirs and files (no Streamlit calls except warnings)."""
    try:
        entries = list(path.iterdir())
    except PermissionError:
        return None  # signal permission error
    except FileNotFoundError:
        return None  # signal missing

    dirs = sorted([e for e in entries if e.is_dir() and not e.name.startswith(".")], key=lambda x: x.name.lower())
    files = []
    if include_files:
        all_files = [e for e in entries if e.is_file() and not e.name.startswith(".")]
        if patterns:
            import fnmatch

            matched = []
            for e in all_files:
                for pat in patterns:
                    if fnmatch.fnmatch(e.name, pat):
                        matched.append(e)
                        break
            # Track whether filter had any hits; don't silently hide the fact that nothing matched.
            # The caller decides how to render the empty-match case via the returned flag.
            files_all = sorted(all_files, key=lambda x: x.name.lower())
            files_matched = sorted(matched, key=lambda x: x.name.lower())
            return dirs, files_matched, files_all
        else:
            files = sorted(all_files, key=lambda x: x.name.lower())
            return dirs, files, files
    return dirs, [], []


def _list_dir(path: Path, include_dirs: bool, include_files: bool, patterns: Optional[List[str]] = None):
    """Public dir listing with pattern-aware fallback + user-visible warnings."""
    path_str = str(path.resolve())
    patterns_key = "|".join(patterns) if patterns else ""
    result = _cached_list_dir(path_str, include_dirs, include_files, patterns_key)
    if result is None:
        # Distinguish permission vs missing by try
        if not path.exists():
            st.warning(f"Directory not found: {path}")
        else:
            st.warning(f"Permission denied: {path}")
        return [], []

    # Unpack pattern-aware triple when present
    if isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], list):
        dirs, files_matched, files_all = result
        # Only for file pickers with patterns
        if patterns and include_files:
            if not files_matched and files_all:
                # No file matched the patterns – show everything but tell the user
                st.caption(f"_No files matching `{patterns_key}` — showing all {len(files_all)} files._")
                return dirs, files_all
            return dirs, files_matched
        # Non-pattern case already handled
        return dirs, files_matched

    # Back-compat: older cache entries or non-pattern paths return (dirs, files)
    dirs, files = result[0], result[1]
    return dirs, files


def _breadcrumb(current: Path, root: Path, state_key: str):
    """Render breadcrumb navigation. Clicking a crumb navigates there."""
    # Build chain from root to current
    try:
        rel = current.resolve().relative_to(root.resolve())
        parts = rel.parts
    except ValueError:
        parts = []
        # Outside root – show absolute
        st.caption(f"📍 `{current}`")
        return

    # Compact mode for deep paths: avoid creating too many Streamlit columns
    if len(parts) + 1 > _MAX_BREADCRUMB_COLS:
        # Show as a single markdown line with the full path, plus a "Jump" selectbox
        st.caption(f"📍 `{_to_repo_rel(current) or '.'}`")
        # Build options for a compact jump
        cur = root.resolve()
        options = [(str(root.resolve()), _to_repo_rel(root) or root.name or "/")]
        for part in parts:
            cur = cur / part
            options.append((str(cur), _to_repo_rel(cur)))
        # Select current is last
        labels = [lbl for _, lbl in options]
        paths = [p for p, _ in options]
        # Map label -> path; handle duplicates by using path as option
        choice = st.selectbox("Jump to", options=paths, format_func=lambda p: _to_repo_rel(Path(p)) or p,
                              index=len(paths) - 1, key=f"{state_key}_breadcrumb_jump",
                              label_visibility="collapsed")
        if choice and Path(choice) != current.resolve():
            st.session_state[state_key + "_cwd"] = choice
            st.rerun()
        return

    cols = st.columns([1] * (len(parts) + 1 + 1))  # root + parts + spacer
    # Root crumb
    with cols[0]:
        label = f"📁 {root.name or '/'}"
        if current.resolve() != root.resolve():
            if st.button(label, key=f"{state_key}_crumb_root", use_container_width=True):
                st.session_state[state_key + "_cwd"] = str(root.resolve())
                st.rerun()
        else:
            st.button(label, key=f"{state_key}_crumb_root_active", disabled=True, use_container_width=True)

    cur = root.resolve()
    for idx, part in enumerate(parts):
        cur = cur / part
        with cols[idx + 1]:
            # Last crumb is current dir – disabled
            is_last = idx == len(parts) - 1
            if is_last:
                st.button(f"📁 {part}", key=f"{state_key}_crumb_{idx}_active", disabled=True, use_container_width=True)
            else:
                if st.button(f"📁 {part}", key=f"{state_key}_crumb_{idx}", use_container_width=True):
                    st.session_state[state_key + "_cwd"] = str(cur)
                    st.rerun()


def _init_cwd(state_key: str, root: Path, initial: str = ""):
    """Initialize current working directory in session state."""
    cwd_key = state_key + "_cwd"
    sel_key = state_key + "_selected"
    init_key = state_key + "_cwd_init"

    # Record what initial was last used to initialize cwd, so we can detect changes
    # (e.g. experiment name changed -> embeddings dir changed). If initial changed,
    # we should re-derive cwd instead of keeping the stale directory.
    if cwd_key not in st.session_state:
        # First init – derive from initial
        _derive_cwd_from_initial(state_key, root, initial)
        st.session_state[init_key] = initial
    elif st.session_state.get(init_key) != initial:
        # initial changed (e.g. experiment/layer switched) – update cwd to follow
        _derive_cwd_from_initial(state_key, root, initial)
        st.session_state[init_key] = initial

    if sel_key not in st.session_state:
        st.session_state[sel_key] = initial.strip() if initial else ""


def _derive_cwd_from_initial(state_key: str, root: Path, initial: str):
    """Set <key>_cwd based on initial path (its parent directory if initial is a file)."""
    cwd_key = state_key + "_cwd"
    if initial:
        try:
            p = Path(initial)
            if not p.is_absolute():
                p = _repo_root() / p
            # If initial is a file, start at its parent; if dir, start there
            candidate = p.parent if p.is_file() or p.suffix else p
            if candidate.exists() and candidate.is_dir():
                try:
                    candidate.resolve().relative_to(root.resolve())
                    st.session_state[cwd_key] = str(candidate.resolve())
                    return
                except ValueError:
                    pass
        except Exception:
            pass
        st.session_state[cwd_key] = str(root.resolve())
    else:
        st.session_state[cwd_key] = str(root.resolve())


def _filter_entries(entries: List[Path], query: str) -> List[Path]:
    """Case-insensitive substring filter."""
    if not query:
        return entries
    q = query.lower().strip()
    return [e for e in entries if q in e.name.lower()]


def _paginate(entries: List[Path], page: int, page_size: int) -> tuple[List[Path], int, int]:
    """Return (page_entries, total_pages, clamped_page)."""
    total = len(entries)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    start = page * page_size
    end = start + page_size
    return entries[start:end], total_pages, page


# ---------------------------------------------------------------------------
# Public: file_picker
# ---------------------------------------------------------------------------

def file_picker(
    label: str,
    key: str,
    root: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    placeholder: str = "",
    help: Optional[str] = None,
    initial: str = "",
) -> str:
    """Browse and pick a file. Returns repo-relative path string.

    Args:
        label: Widget label shown above the picker.
        key: Unique Streamlit key prefix.
        root: Browse root directory (repo-relative or absolute). Defaults to repo root.
        patterns: Glob patterns to filter files (e.g. ["*.csv", "*.tsv"]).
                  Shown first; all files shown if nothing matches.
        placeholder: Shown when nothing selected.
        help: Help text for the label.
        initial: Initial selected value (repo-relative or absolute).

    Returns:
        Selected path as repo-relative string ("" if none).
    """
    root_path = _normalize_root(root)
    # Prefer persisted selection over passed initial (unless initial explicitly changed via _init_cwd)
    stored = st.session_state.get(key + "_selected", None)
    current_initial = stored if stored is not None else (initial or "")

    _init_cwd(key, root_path, initial=current_initial)

    cwd_key = key + "_cwd"
    sel_key = key + "_selected"
    manual_key = key + "_manual"
    show_key = key + "_show_browser"
    page_key = key + "_page"
    filter_key = key + "_filter"
    cwd = Path(st.session_state[cwd_key])

    # Clamp cwd to within root
    try:
        cwd.resolve().relative_to(root_path.resolve())
    except ValueError:
        cwd = root_path
        st.session_state[cwd_key] = str(cwd)

    st.markdown(f"**{label}**" + (f"  \n*{help}*" if help else ""))

    # Current selection display + manual fallback
    sel = st.session_state.get(sel_key, "")
    col_disp, col_clear = st.columns([5, 1])
    with col_disp:
        if sel:
            exists = _path_exists(sel)
            if exists:
                st.caption(f"✅ Selected: `{sel}`")
            else:
                st.caption(f"⚠️ Selected (not found): `{sel}`")
        elif placeholder:
            st.caption(f"_{placeholder}_")
    with col_clear:
        if sel and st.button("Clear", key=key + "_clear", use_container_width=True):
            st.session_state[sel_key] = ""
            st.session_state[show_key] = True  # re-expand so user can pick again
            st.session_state[page_key] = 0
            # Keep manual widget in sync (it otherwise retains the old string)
            if manual_key in st.session_state:
                st.session_state[manual_key] = ""
            st.rerun()

    # Manual path input (fallback for power users / copy-paste)
    manual = st.text_input(
        "Path (or browse below)",
        value=sel,
        key=manual_key,
        placeholder=placeholder,
        label_visibility="collapsed",
    )
    # If user edited manual input, propagate immediately (on each rerun)
    if manual != sel:
        st.session_state[sel_key] = manual.strip()
        # If manual points to an existing file's directory, jump browser there
        if manual.strip():
            try:
                p = Path(manual.strip())
                if not p.is_absolute():
                    p = _repo_root() / p
                if p.exists():
                    target = p.parent if p.is_file() else p
                    if target.is_dir():
                        try:
                            target.resolve().relative_to(root_path.resolve())
                            st.session_state[cwd_key] = str(target.resolve())
                        except ValueError:
                            pass
            except Exception:
                pass
        # Keep display variable in sync for this run
        sel = manual.strip()
        if sel and not _path_exists(sel):
            st.caption(f"⚠️ Path does not exist: `{sel}` — check spelling or browse to select.")

    # Collapsible browser: default expanded when nothing selected, auto-collapse after selection.
    if show_key not in st.session_state:
        st.session_state[show_key] = (sel == "")

    show_browser = bool(st.session_state[show_key])
    toggle_label = "🔼 Hide browser" if show_browser else "📂 Browse files"
    if st.button(toggle_label, key=key + "_toggle_browser", use_container_width=True):
        st.session_state[show_key] = not show_browser
        st.rerun()
        show_browser = not show_browser

    if show_browser:
        with _bordered_container():
            _breadcrumb(cwd, root_path, key)

            # Parent button
            if cwd.resolve() != root_path.resolve():
                if st.button("⬆️ Parent directory", key=key + "_parent", use_container_width=True):
                    parent = cwd.parent
                    # Don't go above root
                    try:
                        parent.resolve().relative_to(root_path.resolve())
                        st.session_state[cwd_key] = str(parent.resolve())
                    except ValueError:
                        st.session_state[cwd_key] = str(root_path.resolve())
                    st.session_state[page_key] = 0
                    st.rerun()

            dirs, files = _list_dir(cwd, include_dirs=True, include_files=True, patterns=patterns)

            if not dirs and not files:
                st.caption("_Empty directory_")

            # ---- Search filter ----
            filter_q = st.text_input(
                "🔍 Filter", key=filter_key, placeholder="Type to filter by name…",
                label_visibility="collapsed",
            )
            # Apply filter to both dirs and files (dirs are usually few, but filter them too)
            f_dirs = _filter_entries(dirs, filter_q)

            # Separate pagination: keep all dirs visible (navigation), only paginate files.
            # This avoids the previous bug where dirs were merged into the paginated list
            # and disappeared on later pages when dirs+files overflowed one page.
            f_files = _filter_entries(files, filter_q)

            # Render dirs (always fully, they are few)
            if f_dirs or not filter_q:
                for d in f_dirs:
                    if st.button(f"📁 {d.name}/", key=_entry_key(key + "_dir", d, cwd), use_container_width=True):
                        st.session_state[cwd_key] = str(d.resolve())
                        st.session_state[page_key] = 0
                        st.session_state[filter_key] = ""  # clear filter on navigation
                        st.rerun()
            if filter_q and not f_dirs and dirs:
                st.caption(f"_No dirs matching `{filter_q}`_")

            # Files: paginated
            page = int(st.session_state.get(page_key, 0))
            page_entries, total_pages, page = _paginate(f_files, page, _PAGE_SIZE)
            st.session_state[page_key] = page

            if filter_q and not f_files:
                if files:
                    st.caption(f"_No files matching `{filter_q}`_")
            elif len(f_files) > _PAGE_SIZE:
                st.caption(f"Showing {len(page_entries)} of {len(f_files)} files — page {page+1}/{total_pages}")

            # Pagination controls (top)
            if total_pages > 1:
                pc1, pc2, pc3 = st.columns([1, 2, 1])
                with pc1:
                    if st.button("◀ Prev", key=key + "_prev", use_container_width=True, disabled=(page == 0)):
                        st.session_state[page_key] = max(0, page - 1)
                        st.rerun()
                with pc3:
                    if st.button("Next ▶", key=key + "_next", use_container_width=True, disabled=(page >= total_pages - 1)):
                        st.session_state[page_key] = min(total_pages - 1, page + 1)
                        st.rerun()
                with pc2:
                    st.caption(f"Page {page+1} / {total_pages}")

            # Render current page entries (files only; dirs already rendered above)
            for f in page_entries:
                is_selected = bool(sel and _to_repo_rel(f) == sel)
                label_text = f"{'✅ ' if is_selected else '📄 '}{f.name}"
                if st.button(label_text, key=_entry_key(key + "_file", f, cwd), use_container_width=True, disabled=is_selected):
                    rel = _to_repo_rel(f)
                    st.session_state[sel_key] = rel
                    st.session_state[show_key] = False  # auto-collapse after selection
                    st.session_state[page_key] = 0
                    # Keep manual widget in sync
                    st.session_state[manual_key] = rel
                    st.rerun()

            # Pagination controls (bottom, for long pages)
            if total_pages > 1:
                pc1, pc2, pc3 = st.columns([1, 2, 1])
                with pc1:
                    if st.button("◀ Prev", key=key + "_prev2", use_container_width=True, disabled=(page == 0)):
                        st.session_state[page_key] = max(0, page - 1)
                        st.rerun()
                with pc3:
                    if st.button("Next ▶", key=key + "_next2", use_container_width=True, disabled=(page >= total_pages - 1)):
                        st.session_state[page_key] = min(total_pages - 1, page + 1)
                        st.rerun()

            st.caption(f"Browsing: `{_to_repo_rel(cwd) or '.'}` — {len(dirs)} dirs, {len(f_files) if filter_q else len(files)} files"
                       + (f" (filtered: {len(f_dirs)} dirs, {len(f_files)} files)" if filter_q else ""))

    return st.session_state.get(sel_key, "").strip()


def dir_picker(
    label: str,
    key: str,
    root: Optional[str] = None,
    help: Optional[str] = None,
    initial: str = "",
) -> str:
    """Browse and pick a directory. Returns repo-relative path string."""
    root_path = _normalize_root(root)
    stored = st.session_state.get(key + "_selected", None)
    current_initial = stored if stored is not None else (initial or "")
    _init_cwd(key, root_path, initial=current_initial)

    cwd_key = key + "_cwd"
    sel_key = key + "_selected"
    manual_key = key + "_manual"
    show_key = key + "_show_browser"
    page_key = key + "_page_dir"
    filter_key = key + "_filter_dir"
    cwd = Path(st.session_state[cwd_key])

    try:
        cwd.resolve().relative_to(root_path.resolve())
    except ValueError:
        cwd = root_path
        st.session_state[cwd_key] = str(cwd)

    st.markdown(f"**{label}**" + (f"  \n*{help}*" if help else ""))

    sel = st.session_state.get(sel_key, "")
    col_disp, col_clear = st.columns([5, 1])
    with col_disp:
        if sel:
            exists = _path_exists(sel)
            if exists:
                st.caption(f"✅ Selected: `{sel}`")
            else:
                st.caption(f"⚠️ Selected (not found): `{sel}`")
        else:
            st.caption("_No directory selected_")
    with col_clear:
        if sel and st.button("Clear", key=key + "_clear", use_container_width=True):
            st.session_state[sel_key] = ""
            st.session_state[show_key] = True
            st.session_state[page_key] = 0
            if manual_key in st.session_state:
                st.session_state[manual_key] = ""
            st.rerun()

    # Manual input + select current button
    col_manual, col_select = st.columns([4, 1])
    with col_manual:
        manual = st.text_input(
            "Path (or browse below)",
            value=sel,
            key=manual_key,
            placeholder="e.g. Outputs/my_experiment/checkpoints/best",
            label_visibility="collapsed",
        )
        if manual != sel:
            st.session_state[sel_key] = manual.strip()
            sel = manual.strip()
            if sel and not _path_exists(sel):
                st.caption(f"⚠️ Path does not exist: `{sel}` — browse to select.")
    with col_select:
        if st.button("Use current", key=key + "_use_current", use_container_width=True, help="Select the currently browsed directory"):
            st.session_state[sel_key] = _to_repo_rel(cwd)
            st.session_state[show_key] = False  # auto-collapse after selection
            if manual_key in st.session_state:
                st.session_state[manual_key] = _to_repo_rel(cwd)
            st.rerun()

    if show_key not in st.session_state:
        st.session_state[show_key] = (sel == "")

    show_browser = bool(st.session_state[show_key])
    toggle_label = "🔼 Hide browser" if show_browser else "📂 Browse directories"
    if st.button(toggle_label, key=key + "_toggle_browser", use_container_width=True):
        st.session_state[show_key] = not show_browser
        st.rerun()
        show_browser = not show_browser

    if show_browser:
        with _bordered_container():
            _breadcrumb(cwd, root_path, key)

            if cwd.resolve() != root_path.resolve():
                if st.button("⬆️ Parent directory", key=key + "_dir_parent", use_container_width=True):
                    parent = cwd.parent
                    try:
                        parent.resolve().relative_to(root_path.resolve())
                        st.session_state[cwd_key] = str(parent.resolve())
                    except ValueError:
                        st.session_state[cwd_key] = str(root_path.resolve())
                    st.session_state[page_key] = 0
                    st.rerun()

            dirs, _ = _list_dir(cwd, include_dirs=True, include_files=False)

            if not dirs:
                st.caption("_No subdirectories_")

            # Filter + pagination for directories too (Outputs/ can have many experiments)
            filter_q = st.text_input(
                "🔍 Filter", key=filter_key, placeholder="Type to filter by name…",
                label_visibility="collapsed",
            )
            f_dirs = _filter_entries(dirs, filter_q)
            page = int(st.session_state.get(page_key, 0))
            page_entries, total_pages, page = _paginate(f_dirs, page, _PAGE_SIZE)
            st.session_state[page_key] = page

            if filter_q and not f_dirs:
                st.caption(f"_No matches for `{filter_q}`_")
            elif len(f_dirs) > _PAGE_SIZE:
                st.caption(f"Showing {len(page_entries)} of {len(f_dirs)} dirs — page {page+1}/{total_pages}")

            if total_pages > 1:
                pc1, pc2, pc3 = st.columns([1, 2, 1])
                with pc1:
                    if st.button("◀ Prev", key=key + "_dir_prev", use_container_width=True, disabled=(page == 0)):
                        st.session_state[page_key] = max(0, page - 1)
                        st.rerun()
                with pc3:
                    if st.button("Next ▶", key=key + "_dir_next", use_container_width=True, disabled=(page >= total_pages - 1)):
                        st.session_state[page_key] = min(total_pages - 1, page + 1)
                        st.rerun()
                with pc2:
                    st.caption(f"Page {page+1} / {total_pages}")

            for d in page_entries:
                is_selected = bool(sel and _to_repo_rel(d) == sel)
                label_text = f"{'✅ ' if is_selected else '📁 '}{d.name}/"
                c1, c2 = st.columns([4, 1])
                with c1:
                    if st.button(label_text, key=_entry_key(key + "_dir_nav", d, cwd), use_container_width=True):
                        st.session_state[cwd_key] = str(d.resolve())
                        st.session_state[page_key] = 0
                        st.session_state[filter_key] = ""
                        st.rerun()
                with c2:
                    if st.button("Select", key=_entry_key(key + "_dir_sel", d, cwd), use_container_width=True, disabled=is_selected):
                        st.session_state[sel_key] = _to_repo_rel(d)
                        st.session_state[show_key] = False  # auto-collapse after selection
                        if manual_key in st.session_state:
                            st.session_state[manual_key] = _to_repo_rel(d)
                        st.rerun()

            if total_pages > 1:
                pc1, pc2, pc3 = st.columns([1, 2, 1])
                with pc1:
                    if st.button("◀ Prev", key=key + "_dir_prev2", use_container_width=True, disabled=(page == 0)):
                        st.session_state[page_key] = max(0, page - 1)
                        st.rerun()
                with pc3:
                    if st.button("Next ▶", key=key + "_dir_next2", use_container_width=True, disabled=(page >= total_pages - 1)):
                        st.session_state[page_key] = min(total_pages - 1, page + 1)
                        st.rerun()

            st.caption(f"Browsing: `{_to_repo_rel(cwd) or '.'}` — {len(dirs)} subdirectories"
                       + (f" (filtered: {len(f_dirs)} dirs)" if filter_q else ""))

    return st.session_state.get(sel_key, "").strip()
