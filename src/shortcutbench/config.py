"""Load the single YAML config."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
DEFAULT_HOME = Path.home() / ".cache" / "shortcut-xai-bench"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the repo root so scripts work from any cwd.
    os.environ.setdefault("SHORTCUT_BENCH_HOME", str(DEFAULT_HOME))
    for key, val in cfg["paths"].items():
        p = Path(os.path.expandvars(os.path.expanduser(val)))
        cfg["paths"][key] = str(p if p.is_absolute() else REPO_ROOT / p)
    if "wandb" in cfg and cfg["wandb"].get("dir"):
        cfg["wandb"]["dir"] = os.path.expandvars(os.path.expanduser(cfg["wandb"]["dir"]))
    return cfg
