#!/usr/bin/env python3
"""CrossPLM unified CLI.

Run from the repository root (no cd needed, no install required):

  python crossplm.py training init --task_name my_experiment
  python crossplm.py training train --config Outputs/my_experiment/config.yaml
  python crossplm.py training eval --checkpoint ... --csv Dataset/mBMRB.csv
  python crossplm.py training labelmap --name my_dataset
  python crossplm.py single extract_embeddings ...
  python crossplm.py single train_sae ...
  python crossplm.py single analyze_features ...
  python crossplm.py single analyze_concepts build|align|heldout ...
  python crossplm.py single analyze_sequence ...
  python crossplm.py single analyze_coactivation ...
  python crossplm.py single evaluate_fidelity ...
  python crossplm.py single evaluate_intervention ...
  python crossplm.py single visualize_features ...
  python crossplm.py crossing compute_feature_similarity ...
  python crossplm.py crossing cross_task_probe ...
  python crossplm.py crossing classify_features ...

If the package is installed (pip install -e .), the same commands work as the
bare `crossplm` command:  `crossplm training eval ...`, `crossplm single ...`.
"""
import importlib
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))

_SINGLE_SCRIPTS = [
    "extract_embeddings",
    "train_sae",
    "analyze_features",
    "analyze_concepts",
    "analyze_sequence",
    "analyze_coactivation",
    "evaluate_fidelity",
    "evaluate_intervention",
    "visualize_features",
]

_CROSSING_SCRIPTS = [
    "compute_feature_similarity",
    "cross_task_probe",
    "classify_features",
]


def _ensure_importable():
    for d in ("Training", "Single", "Crossing"):
        p = os.path.join(_REPO, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _usage():
    print("CrossPLM CLI")
    print("  python crossplm.py training {init,labelmap,train,eval} ...")
    print("  python crossplm.py single {%s} ..." % ",".join(_SINGLE_SCRIPTS))
    print("  python crossplm.py crossing {%s} ..." % ",".join(_CROSSING_SCRIPTS))


def main():
    _ensure_importable()
    argv = sys.argv[1:]
    if not argv:
        _usage()
        sys.exit(1)

    if argv[0] == "training":
        from training_cli import main as training_main
        training_main(argv[1:])
    elif argv[0] == "single":
        if len(argv) < 2:
            print("Usage: crossplm single <command> ...  "
                  f"(commands: {', '.join(_SINGLE_SCRIPTS)})")
            sys.exit(1)
        script = argv[1]
        if script not in _SINGLE_SCRIPTS:
            print(f"Unknown single command '{script}'. "
                  f"Available: {', '.join(_SINGLE_SCRIPTS)}")
            sys.exit(1)
        mod = importlib.import_module(f"single.scripts.{script}")
        # Re-point sys.argv so the delegated argparse shows the right prog name.
        sys.argv = [f"crossplm single {script}"] + argv[2:]
        mod.main()
    elif argv[0] == "crossing":
        if len(argv) < 2:
            print("Usage: crossplm crossing <command> ...  "
                  f"(commands: {', '.join(_CROSSING_SCRIPTS)})")
            sys.exit(1)
        script = argv[1]
        if script not in _CROSSING_SCRIPTS:
            print(f"Unknown crossing command '{script}'. "
                  f"Available: {', '.join(_CROSSING_SCRIPTS)}")
            sys.exit(1)
        mod = importlib.import_module(f"crossing.scripts.{script}")
        sys.argv = [f"crossplm crossing {script}"] + argv[2:]
        mod.main()
    else:
        print(f"Unknown module '{argv[0]}'. Use 'training', 'single', or 'crossing'.")
        _usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
