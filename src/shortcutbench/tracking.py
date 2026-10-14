"""Optional Weights & Biases logging. Everything is a no-op unless enabled.

W&B layout:
  group     = patch_type        (compare obvious vs subtle)
  job_type  = model             (cnn / vit)
  config    = patch_type, rate, model, seed, epochs, ... -> "Group by rate" in the UI
  run id    = hash of the run's full training key, with resume="allow", so a
              rerun updates the same W&B run instead of duplicating it.
Runs loaded from the checkpoint cache replay their stored history.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .scoring import ALL_METRICS
from .train import run_key, run_paths, train_run

log = logging.getLogger(__name__)

SUMMARY_KEYS = ("acc_poisoned", "acc_clean", "acc_poisoned_target", "acc_clean_target",
                "reliance_gap", "trigger_flip_rate", "clean_nontarget_called_target",
                "valid", "final_val_acc", "train_sec", "sec_per_epoch", "n_params")


def _wandb():
    import wandb   # imported lazily so offline runs never need it configured
    return wandb


class NullTracker:
    def log_epoch(self, row: dict) -> None: ...
    def log(self, data: dict) -> None: ...
    def finish(self, rec: dict | None = None) -> None: ...


def wandb_run_id(cfg: dict, patch_type: str, rate: float, model: str, seed: int, epochs: int) -> str:
    key = run_key(cfg, patch_type, rate, model, seed, epochs)
    return run_paths(cfg, key)[0].stem.split("_")[-1] + f"-{patch_type}-{model}"


class RunTracker(NullTracker):
    def __init__(self, cfg: dict, patch_type: str, rate: float, model: str, seed: int, epochs: int):
        wandb = _wandb()
        key = run_key(cfg, patch_type, rate, model, seed, epochs)
        run_id = wandb_run_id(cfg, patch_type, rate, model, seed, epochs)
        wcfg = cfg["wandb"]
        Path(wcfg["dir"]).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=wcfg["project"], entity=wcfg.get("entity"), dir=wcfg["dir"],
            id=run_id, resume="allow", reinit="finish_previous",
            name=f"{patch_type}-r{rate:.1f}-{model}-s{seed}",
            group=patch_type, job_type=model,
            tags=[patch_type, model, f"rate={rate:.1f}", f"seed={seed}"],
            config={"patch_type": patch_type, "rate": rate, "model": model, "seed": seed,
                    "epochs": epochs, "spec": key["spec"], "model_cfg": key["model_cfg"],
                    "opt_cfg": key["opt_cfg"], "batch_size": key["batch_size"],
                    "augment": key["augment"]},
        )
        self.n_logged = 0

    def log_epoch(self, row: dict) -> None:
        self.run.log({f"train/{k}" if k != "epoch" else k: v for k, v in row.items()}, step=row["epoch"])
        self.n_logged += 1

    def log(self, data: dict) -> None:
        self.run.log(data)

    def finish(self, rec: dict | None = None) -> None:
        if rec is not None:
            if self.n_logged == 0:                      # cached run: replay its history
                for row in rec.get("history", []):
                    self.log_epoch(row)
            for k in SUMMARY_KEYS:
                if k in rec:
                    self.run.summary[k] = rec[k]
            self.run.summary["status"] = "VALID" if rec["valid"] else "INVALID"
        self.run.finish()


def train_tracked(cfg: dict, patch_type: str, rate: float, model: str, seed: int,
                  epochs: int | None = None, use_wandb: bool = False) -> dict:
    """train_run + optional W&B logging of curves and reliance metrics."""
    epochs = int(epochs or cfg["train"][model]["epochs"])
    tracker = RunTracker(cfg, patch_type, rate, model, seed, epochs) if use_wandb else NullTracker()
    try:
        rec = train_run(cfg, patch_type, rate, model, seed, epochs=epochs, log_fn=tracker.log_epoch)
    except BaseException:
        tracker.finish(None)
        raise
    tracker.finish(rec)
    return rec


def log_samples(cfg: dict, patch_type: str, use_wandb: bool) -> None:
    """Log results/samples/<patch_type>/ as a W&B artifact (once; identical content dedups)."""
    if not use_wandb:
        return
    wandb = _wandb()
    d = Path(cfg["paths"]["results_dir"]) / "samples" / patch_type
    if not (d / "grid.png").exists():
        log.warning("no sample grid for %s; run scripts/dump_samples.py first", patch_type)
        return
    wcfg = cfg["wandb"]
    run = wandb.init(project=wcfg["project"], entity=wcfg.get("entity"), dir=wcfg["dir"],
                     id=f"samples-{patch_type}", resume="allow", reinit="finish_previous",
                     name=f"samples-{patch_type}", group=patch_type, job_type="samples",
                     config={"patch_type": patch_type})
    art = wandb.Artifact(f"samples-{patch_type}", type="samples",
                         description=f"injection samples + masks for patch_type={patch_type}")
    art.add_dir(str(d))
    run.log_artifact(art)
    run.log({"samples/grid": wandb.Image(str(d / "grid.png"))})
    run.finish()


def log_scores(cfg: dict, rows: list[dict], figures: list[Path] | None = None) -> None:
    """Attach detection scores to each training run's W&B entry (same run id,
    resumed), and log the tidy table + figures to one summary run."""
    wandb = _wandb()
    wcfg = cfg["wandb"]
    by_run: dict[tuple, dict] = {}
    for r in rows:
        k = (r["patch_type"], r["rate"], r["model"], r["seed"])
        by_run.setdefault(k, {})
        for m in ALL_METRICS:
            by_run[k][f"score/{r['method']}/{m}"] = r[m]
    for (pt, rate, model, seed), summary in by_run.items():
        rid = wandb_run_id(cfg, pt, rate, model, seed, cfg["train"][model]["epochs"])
        run = wandb.init(project=wcfg["project"], entity=wcfg.get("entity"), dir=wcfg["dir"], id=rid,
                         resume="allow", reinit="finish_previous")
        for k, v in summary.items():
            run.summary[k] = v
        run.finish()
    run = wandb.init(project=wcfg["project"], entity=wcfg.get("entity"), dir=wcfg["dir"], id="sweep-summary",
                     resume="allow", reinit="finish_previous", name="sweep-summary", job_type="summary")
    cols = list(rows[0])
    run.log({"results": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
    for f in figures or []:
        run.log({f"figures/{Path(f).stem}": wandb.Image(str(f))})
    run.finish()
