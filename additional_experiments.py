"""Confirmation experiments for the optimizer benchmark.

This module leaves the verified ``results/`` directory untouched and writes a
separate, downloadable ``results_additional/`` bundle.  The full workflow expands
and replicates scheduler tuning, increases held-out evaluation coverage, and tests
the width-4096 learning-rate extrapolation.  A small CPU smoke profile exercises
the same orchestration without claiming benchmark evidence.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

import transformer_optimizer_benchmark as base


FULL_SEEDS = [42, 314, 2718]
FULL_SCHEDULER_LRS = [6e-4, 1.2e-3, 2.4e-3, 4.8e-3]
FULL_WARMUPS = [5, 10, 20, 40]
FULL_WIDTH_GRIDS = {
    2048: [2.5e-5, 3.5355339059e-5, 5e-5, 7.0710678119e-5, 1e-4],
    4096: [1.25e-5, 1.7677669530e-5, 2.5e-5, 3.5355339059e-5, 5e-5],
}
ADDITIONAL_PLOTS = (
    "scheduler_tuning_additional.png",
    "scheduler_final_losses_additional.png",
    "relative_updates_cosine_additional.png",
    "relative_updates_wsd_additional.png",
    "width_confirmation_additional.png",
)


@dataclass(frozen=True)
class AdditionalTrainConfig:
    width: int
    peak_lr: float
    warmup: int
    steps: int
    planned_steps: int
    schedule: str
    effective_batch_size: int
    micro_batch_size: int
    context: int
    model_seed: int
    data_seed: int
    weight_decay: float = 0.1
    log_ratios: bool = False
    gradient_checkpointing: bool = False
    capture_state: bool = False

    @property
    def accumulation_steps(self) -> int:
        assert self.effective_batch_size % self.micro_batch_size == 0
        return self.effective_batch_size // self.micro_batch_size


class CheckpointedCharacterTransformer(base.CharacterTransformer):
    """The baseline model with optional activation checkpointing."""

    def __init__(self, *args: Any, gradient_checkpointing: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.final_norm(x)
        return self.head_next(x[:, :-1]), self.head_two(x[:, :-2])


def additional_plan(smoke: bool = False) -> dict[str, Any]:
    if smoke:
        return {
            "seeds": [42, 314],
            "scheduler_lrs": [3e-4, 6e-4, 1.2e-3],
            "warmups": [1, 2, 3],
            "scheduler_steps": 4,
            "planned_steps": 6,
            "width_grids": {32: [2e-4, 5e-4, 1e-3], 64: [1e-4, 2.5e-4, 5e-4]},
            "width_steps": 4,
            "width_warmup": 1,
            "context": 16,
            "effective_batch_size": 2,
            "validation_batches": 4,
        }
    return {
        "seeds": FULL_SEEDS,
        "scheduler_lrs": FULL_SCHEDULER_LRS,
        "warmups": FULL_WARMUPS,
        "scheduler_steps": 200,
        "planned_steps": 300,
        "width_grids": FULL_WIDTH_GRIDS,
        "width_steps": 120,
        "width_warmup": 20,
        "context": 128,
        "effective_batch_size": 8,
        "validation_batches": 32,
    }


def estimated_parameter_count(vocab_size: int, width: int, context: int) -> int:
    """Exact count for the fixed four-block, two-head-output architecture."""
    return 48 * width * width + (3 * vocab_size + context + 18) * width


def training_memory_estimate(parameter_count: int) -> int:
    """Persistent fp32 AdamW bytes: parameter, gradient, first and second moments."""
    return 16 * parameter_count


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_controlled(
    config: AdditionalTrainConfig,
    data: base.CharacterData,
    device: torch.device,
    val_batches: list[torch.Tensor],
) -> dict[str, Any]:
    """Train with independently controlled initialization and data-order seeds."""
    base.seed_everything(config.model_seed)
    model = CheckpointedCharacterTransformer(
        len(data.vocab), config.width, config.context,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    # foreach=False avoids a parameter-sized temporary tensor list at width 4096.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.peak_lr, betas=(0.9, 0.95),
        weight_decay=config.weight_decay, foreach=False,
    )
    accumulation = config.accumulation_steps
    batches = base.CharacterData.batches(
        data.train, config.data_seed, config.steps * accumulation,
        config.micro_batch_size, config.context,
    )
    history: list[dict[str, float | int]] = []
    ratio_history: list[dict[str, float | int]] = []
    for step in range(1, config.steps + 1):
        lr = base.schedule_lr(
            config.schedule, step, config.peak_lr, config.warmup,
            config.planned_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        before = None
        if config.log_ratios:
            before = {
                name: [parameter.detach().clone() for parameter in parameters]
                for name, parameters in base.parameter_groups(model).items()
            }
        optimizer.zero_grad(set_to_none=True)
        losses = []
        start = (step - 1) * accumulation
        for batch in batches[start:start + accumulation]:
            loss = base.objective(model, batch.to(device))
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step, "lr": lr, "train_loss": float(np.mean(losses))})
        if before is not None:
            ratio_history.append({"step": step, **base.relative_update_ratios(before, model)})

    # Gradients are no longer needed and consume one full model copy at large widths.
    optimizer.zero_grad(set_to_none=True)
    final_val_loss = base.evaluate(model, val_batches, device)
    result: dict[str, Any] = {
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": final_val_loss,
        "history": history,
        "ratio_history": ratio_history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if config.capture_state:
        result["state_dict"] = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
    return result


def aggregate_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for values, group in sorted(grouped.items()):
        train = np.array([row["train_loss"] for row in group], dtype=float)
        val = np.array([row["val_loss"] for row in group], dtype=float)
        output.append({
            **dict(zip(keys, values, strict=True)),
            "seeds": [row["seed"] for row in group],
            "mean_train_loss": float(np.mean(train)),
            "std_train_loss": float(np.std(train)),
            "mean_val_loss": float(np.mean(val)),
            "std_val_loss": float(np.std(val)),
            "all_finite": bool(np.isfinite(train).all() and np.isfinite(val).all()),
        })
    return output


def select_finite_minimum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in rows if row["all_finite"]]
    if not finite:
        raise RuntimeError("no finite candidate losses")
    return min(finite, key=lambda row: row["mean_val_loss"])


def logged_run(
    logger: base.RunLogger, out_dir: Path, run_id: str, experiment: str,
    config: AdditionalTrainConfig, expected_seconds: float,
    data: base.CharacterData, device: torch.device,
    val_batches: list[torch.Tensor],
) -> dict[str, Any]:
    try:
        return logger.run(
            run_id, experiment, asdict(config), config.model_seed, expected_seconds,
            lambda: train_controlled(config, data, device, val_batches),
        )
    finally:
        # Preserve progress even if Colab disconnects or a large-width run OOMs.
        logger.write(out_dir)


def scheduler_confirmation(
    plan: dict[str, Any], data: base.CharacterData, device: torch.device,
    validation: dict[int, list[torch.Tensor]], logger: base.RunLogger,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, list[dict[str, Any]]] = {}
    best: dict[str, dict[str, Any]] = {}
    for schedule in ("cosine", "wsd"):
        rows = []
        for lr in plan["scheduler_lrs"]:
            for warmup in plan["warmups"]:
                for seed in plan["seeds"]:
                    config = AdditionalTrainConfig(
                        width=256 if plan["context"] == 128 else 32,
                        peak_lr=lr, warmup=warmup,
                        steps=plan["scheduler_steps"], planned_steps=plan["planned_steps"],
                        schedule=schedule, effective_batch_size=plan["effective_batch_size"],
                        micro_batch_size=plan["effective_batch_size"], context=plan["context"],
                        model_seed=seed, data_seed=100_000 + seed,
                    )
                    result = logged_run(
                        logger, out_dir,
                        f"additional-tune-{schedule}-lr{lr:g}-wu{warmup}-seed{seed}",
                        "additional_scheduler_tuning", config,
                        max(0.1, plan["scheduler_steps"] * 0.065),
                        data, device, validation[seed],
                    )
                    rows.append({"lr": lr, "warmup": warmup, "seed": seed,
                                 "data_seed": config.data_seed,
                                 "train_loss": result["final_train_loss"],
                                 "val_loss": result["final_val_loss"]})
                    del result
                    release_cuda()
        candidates[schedule] = rows
        summaries[schedule] = aggregate_rows(rows, ("lr", "warmup"))
        best[schedule] = select_finite_minimum(summaries[schedule])

    finals: dict[str, Any] = {}
    seed_42_states: dict[str, dict[str, torch.Tensor]] = {}
    for schedule in ("cosine", "wsd"):
        chosen = best[schedule]
        final_rows = []
        ratio_runs = []
        for seed in plan["seeds"]:
            config = AdditionalTrainConfig(
                width=256 if plan["context"] == 128 else 32,
                peak_lr=chosen["lr"], warmup=chosen["warmup"],
                steps=plan["scheduler_steps"], planned_steps=plan["planned_steps"],
                schedule=schedule, effective_batch_size=plan["effective_batch_size"],
                micro_batch_size=plan["effective_batch_size"], context=plan["context"],
                model_seed=seed, data_seed=100_000 + seed, log_ratios=True,
                capture_state=seed == base.BASE_SEED,
            )
            result = logged_run(
                logger, out_dir, f"additional-final-{schedule}-seed{seed}",
                "additional_scheduler_final", config,
                max(0.1, plan["scheduler_steps"] * 0.08),
                data, device, validation[seed],
            )
            final_rows.append({"seed": seed, "data_seed": config.data_seed,
                               "train_loss": result["final_train_loss"],
                               "val_loss": result["final_val_loss"]})
            ratio_runs.append({"seed": seed, "rows": result["ratio_history"]})
            if "state_dict" in result:
                seed_42_states[schedule] = result.pop("state_dict")
            del result
            release_cuda()
        summary = aggregate_rows(final_rows, ())[0]
        finals[schedule] = {"runs": final_rows, "summary": summary,
                            "ratio_runs": ratio_runs}

    winner = min(finals, key=lambda key: finals[key]["summary"]["mean_val_loss"])
    lr_values, warmup_values = plan["scheduler_lrs"], plan["warmups"]
    bracketed = {
        schedule: {
            "learning_rate": best[schedule]["lr"] not in (min(lr_values), max(lr_values)),
            "warmup": best[schedule]["warmup"] not in (min(warmup_values), max(warmup_values)),
        }
        for schedule in ("cosine", "wsd")
    }
    return ({"candidates": candidates, "candidate_summaries": summaries,
             "best_configurations": best, "finals": finals,
             "winner": winner, "bracketed": bracketed}, seed_42_states[winner])


def width_confirmation(
    plan: dict[str, Any], data: base.CharacterData, device: torch.device,
    validation: dict[int, list[torch.Tensor]], logger: base.RunLogger,
    out_dir: Path,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    total_memory = torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else 0
    for width, grid in plan["width_grids"].items():
        parameter_count = estimated_parameter_count(len(data.vocab), width, plan["context"])
        memory_estimate = training_memory_estimate(parameter_count)
        micro_batch = plan["effective_batch_size"]
        checkpointing = False
        if width >= 2048:
            micro_batch = 2 if width == 2048 else 1
            checkpointing = width >= 4096
        capability = {
            "parameter_count": parameter_count,
            "persistent_training_bytes_estimate": memory_estimate,
            "device_total_bytes": total_memory,
            "gradient_checkpointing": checkpointing,
            "effective_batch_size": plan["effective_batch_size"],
            "micro_batch_size": micro_batch,
        }
        if total_memory and memory_estimate > 0.90 * total_memory:
            output[str(width)] = {
                "status": "resource_limited",
                "reason": "estimated fp32 AdamW persistent state exceeds 90% of GPU memory",
                "capability": capability, "observations": [],
            }
            continue

        rows = []
        resource_error = None
        for lr in grid:
            for seed in plan["seeds"]:
                config = AdditionalTrainConfig(
                    width=width, peak_lr=lr, warmup=plan["width_warmup"],
                    steps=plan["width_steps"], planned_steps=plan["width_steps"],
                    schedule="warmup_stable",
                    effective_batch_size=plan["effective_batch_size"],
                    micro_batch_size=micro_batch, context=plan["context"],
                    model_seed=seed, data_seed=100_000 + seed,
                    gradient_checkpointing=checkpointing,
                )
                try:
                    result = logged_run(
                        logger, out_dir,
                        f"additional-width-{width}-lr{lr:.8g}-seed{seed}",
                        "additional_width_confirmation", config,
                        max(0.1, plan["width_steps"] * 0.06 * (width / 256) ** 2),
                        data, device, validation[seed],
                    )
                except RuntimeError as exc:
                    message = str(exc)
                    if "out of memory" not in message.lower():
                        raise
                    resource_error = message
                    release_cuda()
                    break
                rows.append({"lr": lr, "seed": seed, "data_seed": config.data_seed,
                             "train_loss": result["final_train_loss"],
                             "val_loss": result["final_val_loss"]})
                del result
                release_cuda()
            if resource_error:
                break
        if resource_error:
            output[str(width)] = {
                "status": "resource_limited", "reason": resource_error,
                "capability": capability, "observations": rows,
            }
            continue
        summaries = aggregate_rows(rows, ("lr",))
        best = select_finite_minimum(summaries)
        output[str(width)] = {
            "status": "complete", "capability": capability,
            "observations": rows, "summaries": summaries,
            "best": best,
            "boundary_minimum": best["lr"] in (min(grid), max(grid)),
        }
    return output


def scaling_fit(width_results: dict[str, Any]) -> dict[str, Any] | None:
    baseline_path = Path("results/metrics.json")
    if not baseline_path.exists():
        return None
    baseline = json.loads(baseline_path.read_text())
    points = {
        int(width): info["empirical_best_lr"]
        for width, info in baseline["width_sweep"]["widths"].items()
    }
    for width, info in width_results.items():
        if info["status"] == "complete":
            points[int(width)] = info["best"]["lr"]
    if len(points) < 4:
        return None
    widths = np.array(sorted(points), dtype=float)
    lrs = np.array([points[int(width)] for width in widths])
    exponent, intercept = np.polyfit(np.log(widths), np.log(lrs), 1)
    fitted = intercept + exponent * np.log(widths)
    observed = np.log(lrs)
    ss_res = float(np.sum((observed - fitted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return {"points": {str(int(k)): v for k, v in sorted(points.items())},
            "exponent": float(exponent), "intercept": float(intercept),
            "r_squared": 1 - ss_res / max(ss_tot, 1e-30)}


def make_additional_plots(metrics: dict[str, Any], out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/optimizer-mpl")
    import matplotlib.pyplot as plt

    scheduler = metrics["scheduler_confirmation"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, schedule in zip(axes, ("cosine", "wsd"), strict=True):
        summaries = scheduler["candidate_summaries"][schedule]
        for warmup in metrics["plan"]["warmups"]:
            rows = [row for row in summaries if row["warmup"] == warmup]
            axis.errorbar([row["lr"] for row in rows], [row["mean_val_loss"] for row in rows],
                          yerr=[row["std_val_loss"] for row in rows], marker="o",
                          capsize=3, label=f"warmup {warmup}")
        best = scheduler["best_configurations"][schedule]
        axis.scatter([best["lr"]], [best["mean_val_loss"]], marker="*", s=220,
                     color="black", label="selected minimum")
        axis.set_xscale("log")
        axis.set(title=schedule.upper(), xlabel="peak learning rate")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("mean held-out loss across seeds")
    fig.suptitle("Expanded, replicated scheduler tuning")
    fig.tight_layout()
    fig.savefig(out_dir / ADDITIONAL_PLOTS[0], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    labels = ["cosine", "wsd"]
    x = np.arange(2)
    train = [scheduler["finals"][key]["summary"]["mean_train_loss"] for key in labels]
    val = [scheduler["finals"][key]["summary"]["mean_val_loss"] for key in labels]
    train_std = [scheduler["finals"][key]["summary"]["std_train_loss"] for key in labels]
    val_std = [scheduler["finals"][key]["summary"]["std_val_loss"] for key in labels]
    axis.bar(x - .18, train, .36, yerr=train_std, capsize=4, label="train")
    axis.bar(x + .18, val, .36, yerr=val_std, capsize=4, label="held-out")
    axis.set(xticks=x, xticklabels=[label.upper() for label in labels], ylabel="loss",
             title="Three-seed final comparison")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / ADDITIONAL_PLOTS[1], dpi=160)
    plt.close(fig)

    for schedule in ("cosine", "wsd"):
        runs = scheduler["finals"][schedule]["ratio_runs"]
        layers = [key for key in runs[0]["rows"][0] if key != "step"]
        fig, axis = plt.subplots(figsize=(9, 5))
        for layer in layers:
            values = np.array([[row[layer] for row in run["rows"]] for run in runs])
            steps = [row["step"] for row in runs[0]["rows"]]
            axis.plot(steps, values.mean(axis=0), label=layer)
            axis.fill_between(steps, values.mean(axis=0) - values.std(axis=0),
                              values.mean(axis=0) + values.std(axis=0), alpha=.12)
        warmup = scheduler["best_configurations"][schedule]["warmup"]
        axis.axvline(warmup, color="black", linestyle="--",
                     label=f"warmup endpoint ({warmup})")
        axis.set_yscale("log")
        axis.set(xlabel="optimizer step", ylabel="update-to-weight ratio",
                 title=f"{schedule.upper()} mean ± seed std.")
        axis.grid(alpha=.25)
        axis.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"relative_updates_{schedule}_additional.png", dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for width, info in metrics["width_confirmation"].items():
        if info["status"] != "complete":
            continue
        rows = info["summaries"]
        axis.errorbar([row["lr"] for row in rows], [row["mean_val_loss"] for row in rows],
                      yerr=[row["std_val_loss"] for row in rows], marker="o", capsize=3,
                      label=f"width {width}")
        axis.scatter([info["best"]["lr"]], [info["best"]["mean_val_loss"]],
                     marker="*", s=220)
    axis.axvline(2.5e-5, color="black", linestyle="--", label="original 4096 prediction")
    axis.set_xscale("log")
    axis.set(xlabel="learning rate", ylabel="mean held-out loss across seeds",
             title="Large-width LR confirmation")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / ADDITIONAL_PLOTS[4], dpi=160)
    plt.close(fig)


def validate_additional_metrics(metrics: dict[str, Any]) -> None:
    assert metrics["provenance"]["result_kind"] in {
        "colab_cuda_additional", "local_cpu_additional_smoke"
    }
    plan = metrics["plan"]
    expected_candidates = len(plan["scheduler_lrs"]) * len(plan["warmups"]) * len(plan["seeds"])
    scheduler = metrics["scheduler_confirmation"]
    for schedule in ("cosine", "wsd"):
        assert len(scheduler["candidates"][schedule]) == expected_candidates
        assert len(scheduler["finals"][schedule]["runs"]) == len(plan["seeds"])
        assert len(scheduler["finals"][schedule]["ratio_runs"]) == len(plan["seeds"])
        assert all(len(run["rows"]) == plan["scheduler_steps"]
                   for run in scheduler["finals"][schedule]["ratio_runs"])
    assert scheduler["winner"] in {"cosine", "wsd"}
    for width, grid in plan["width_grids"].items():
        info = metrics["width_confirmation"][str(width)]
        assert info["status"] in {"complete", "resource_limited"}
        if info["status"] == "complete":
            assert len(info["observations"]) == len(grid) * len(plan["seeds"])


def write_summary(metrics: dict[str, Any], out_dir: Path) -> None:
    scheduler = metrics["scheduler_confirmation"]
    lines = [
        "# Additional optimizer confirmation results", "",
        "This file is generated from `metrics.json`; it does not replace the reviewed baseline README.", "",
        "## Scheduler comparison", "",
        "| Schedule | Peak LR | Warmup | Mean train loss | Mean held-out loss | Held-out std. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for schedule in ("cosine", "wsd"):
        best = scheduler["best_configurations"][schedule]
        final = scheduler["finals"][schedule]["summary"]
        lines.append(
            f"| {schedule.upper()} | {best['lr']:.6g} | {best['warmup']} | "
            f"{final['mean_train_loss']:.6f} | {final['mean_val_loss']:.6f} | "
            f"{final['std_val_loss']:.6f} |"
        )
    lines += ["", f"Retained scheduler: **{scheduler['winner'].upper()}**.", "",
              "## Large-width confirmation", "",
              "| Width | Status | Selected LR | Mean held-out loss | Boundary? |",
              "|---:|---|---:|---:|:---:|"]
    for width, info in metrics["width_confirmation"].items():
        if info["status"] == "complete":
            lines.append(f"| {width} | complete | {info['best']['lr']:.6g} | "
                         f"{info['best']['mean_val_loss']:.6f} | "
                         f"{'yes' if info['boundary_minimum'] else 'no'} |")
        else:
            lines.append(f"| {width} | resource limited | — | — | — |")
            lines += ["", f"Resource note for width {width}: `{info['reason']}`"]
    lines += ["", "## Interpretation gates", ""]
    for schedule, flags in scheduler["bracketed"].items():
        lines.append(f"- {schedule.upper()} LR bracketed: **{flags['learning_rate']}**; "
                     f"warmup bracketed: **{flags['warmup']}**.")
    fit = metrics.get("scaling_fit")
    if fit:
        lines.append(f"- Combined width exponent: **{fit['exponent']:.4f}** "
                     f"with R² **{fit['r_squared']:.4f}**.")
    lines += ["- Treat any boundary minimum or resource-limited width as unfinished evidence.", "",
              "See `run_log.csv` and the PNG plots for the full audit trail."]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def run_additional(smoke: bool = False, out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or Path("additional_smoke_results" if smoke else "results_additional")
    if smoke:
        assert out_dir.name != "results_additional"
        device = torch.device("cpu")
    else:
        assert torch.cuda.is_available(), "additional full results require a CUDA runtime"
        assert out_dir.name == "results_additional"
        device = torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = additional_plan(smoke)
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    text = base.load_text(Path(".cache/tinyshakespeare.txt"), full=not smoke)
    data = base.CharacterData(text)
    validation = {
        seed: base.CharacterData.batches(
            data.val, 200_000 + seed, plan["validation_batches"],
            plan["effective_batch_size"], plan["context"],
        )
        for seed in plan["seeds"]
    }
    logger = base.RunLogger()
    scheduler, retained = scheduler_confirmation(
        plan, data, device, validation, logger, out_dir,
    )
    torch.save(retained, out_dir / "retained_model.pt")
    del retained
    release_cuda()
    widths = width_confirmation(plan, data, device, validation, logger, out_dir)
    metrics: dict[str, Any] = {
        "provenance": {
            "result_kind": "local_cpu_additional_smoke" if smoke else "colab_cuda_additional",
            "device": "CPU smoke" if smoke else torch.cuda.get_device_name(0),
            "started_at": started_at, "python": platform.python_version(),
            "torch": torch.__version__,
            "source_commit": os.environ.get("SOURCE_COMMIT", "uncommitted"),
            "dataset_sha256": base.DATA_SHA256,
        },
        "plan": plan,
        "scheduler_confirmation": scheduler,
        "width_confirmation": widths,
    }
    metrics["scaling_fit"] = scaling_fit(widths)
    make_additional_plots(metrics, out_dir)
    metrics["timing"] = {
        "total_seconds": time.perf_counter() - wall_start,
        "total_readable": base.readable_duration(time.perf_counter() - wall_start),
        "cuda_synchronized": True,
    }
    metrics["runs"] = logger.records
    validate_additional_metrics(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.write(out_dir)
    write_summary(metrics, out_dir)
    assert all((out_dir / filename).exists() for filename in ADDITIONAL_PLOTS)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run a small CPU orchestration check")
    args = parser.parse_args()
    metrics = run_additional(smoke=args.smoke)
    print(json.dumps({
        "kind": metrics["provenance"]["result_kind"],
        "duration": metrics["timing"]["total_readable"],
        "winner": metrics["scheduler_confirmation"]["winner"],
        "width_status": {width: info["status"]
                         for width, info in metrics["width_confirmation"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
