"""Colab-first transformer optimizer benchmark with auditable timing and result generation.

`run_full()` is the evidence-collection path and refuses to run without a CUDA T4.
It writes machine-readable metrics, logs, plots, and the retained checkpoint, but
never edits README.md. `run_smoke()` exercises the same orchestration on synthetic
text and writes only to `smoke_results/`.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
BASE_SEED = 42
ADAM_GRADIENTS = [0.25, -0.10, 0.40, -0.30, 0.05]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def readable_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


class RunLogger:
    REQUIRED = {
        "run_id", "experiment", "configuration", "seed", "duration_seconds",
        "readable_duration", "final_train_loss", "final_val_loss", "status",
        "peak_gpu_memory_bytes", "expected_seconds",
    }

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def run(
        self, run_id: str, experiment: str, configuration: dict[str, Any],
        seed: int | None, expected_seconds: float, fn: Callable[[], Any],
    ) -> Any:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cuda_sync()
        started = time.perf_counter()
        status, result, error = "ok", None, None
        try:
            result = fn()
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        cuda_sync()
        duration = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        train_loss = result.get("final_train_loss") if isinstance(result, dict) else None
        val_loss = result.get("final_val_loss") if isinstance(result, dict) else None
        record = {
            "run_id": run_id,
            "experiment": experiment,
            "configuration": configuration,
            "seed": seed,
            "duration_seconds": duration,
            "readable_duration": readable_duration(duration),
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "status": status,
            "peak_gpu_memory_bytes": int(peak),
            "expected_seconds": float(expected_seconds),
        }
        if error:
            record["error"] = error
        assert self.REQUIRED <= record.keys()
        self.records.append(record)
        if error:
            raise RuntimeError(f"{run_id} failed: {error}")
        return result

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_log.json").write_text(json.dumps(self.records, indent=2))
        fields = sorted({k for r in self.records for k in r} - {"configuration"}) + ["configuration"]
        with (out_dir / "run_log.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.records:
                cooked = dict(row)
                cooked["configuration"] = json.dumps(cooked["configuration"], sort_keys=True)
                writer.writerow(cooked)


def adam_manual(
    initial_weight: float = 1.0, gradients: Iterable[float] = ADAM_GRADIENTS,
    lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, bias_correction: bool = True,
) -> list[dict[str, float]]:
    w, m, v = float(initial_weight), 0.0, 0.0
    rows = []
    for step, grad in enumerate(gradients, 1):
        g = float(grad)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1**step) if bias_correction else m
        v_hat = v / (1 - beta2**step) if bias_correction else v
        update = lr * m_hat / (math.sqrt(v_hat) + eps)
        w -= update
        rows.append({"step": step, "gradient": g, "m": m, "v": v,
                     "m_hat": m_hat, "v_hat": v_hat, "update": update,
                     "new_weight": w})
    return rows


def adam_torch_reference(initial_weight: float = 1.0) -> list[dict[str, float]]:
    p = nn.Parameter(torch.tensor(initial_weight, dtype=torch.float64))
    opt = torch.optim.Adam([p], lr=0.01, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0.0)
    rows = []
    for step, grad in enumerate(ADAM_GRADIENTS, 1):
        p.grad = torch.tensor(grad, dtype=torch.float64)
        opt.step()
        state = opt.state[p]
        m, v = state["exp_avg"].item(), state["exp_avg_sq"].item()
        mh, vh = m / (1 - 0.9**step), v / (1 - 0.999**step)
        rows.append({"step": step, "gradient": grad, "m": m, "v": v,
                     "m_hat": mh, "v_hat": vh,
                     "update": 0.01 * mh / (math.sqrt(vh) + 1e-8),
                     "new_weight": p.item()})
        opt.zero_grad(set_to_none=True)
    return rows


def assert_adam_parity(rows: list[dict[str, float]], reference: list[dict[str, float]]) -> None:
    for actual, expected in zip(rows, reference, strict=True):
        for key in ("m", "v", "m_hat", "v_hat", "update", "new_weight"):
            assert math.isclose(actual[key], expected[key], rel_tol=2e-13, abs_tol=2e-15), (key, actual, expected)


def bias_correction_diagnostic(max_steps: int = 20_000) -> dict[str, Any]:
    corrected = adam_manual(gradients=(ADAM_GRADIENTS[i % 5] for i in range(max_steps)))
    uncorrected = adam_manual(gradients=(ADAM_GRADIENTS[i % 5] for i in range(max_steps)), bias_correction=False)
    rel = [abs(a["update"] - b["update"]) / max(abs(a["update"]), 1e-15)
           for a, b in zip(corrected, uncorrected)]
    stops = None
    for i in range(len(rel) - 99):
        if max(rel[i:i + 100]) < 0.01:
            stops = i + 1  # one-based first step in the qualifying window
            break
    assert stops is not None
    return {"first_20": [{"step": i + 1,
                           "corrected_weight": corrected[i]["new_weight"],
                           "uncorrected_weight": uncorrected[i]["new_weight"],
                           "corrected_update": corrected[i]["update"],
                           "uncorrected_update": uncorrected[i]["update"],
                           "relative_update_difference": rel[i]}
                          for i in range(20)],
            "stops_mattering_step": stops,
            "definition": "first step beginning 100 consecutive steps with relative update difference < 1%"}


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, context: int):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context, context, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(z: torch.Tensor) -> torch.Tensor:
            return z.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:t, :t], float("-inf"))
        y = scores.softmax(dim=-1) @ v
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, width: int, heads: int, context: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads, context)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width, bias=False), nn.GELU(),
                                 nn.Linear(4 * width, width, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CharacterTransformer(nn.Module):
    def __init__(self, vocab_size: int, width: int, context: int, layers: int = 4, heads: int = 4):
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([Block(width, heads, context) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.head_next = nn.Linear(width, vocab_size, bias=False)
        self.head_two = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = tokens.shape[1]
        x = self.token_embedding(tokens) + self.position_embedding(torch.arange(t, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head_next(x[:, :-1]), self.head_two(x[:, :-2])


def objective(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    one, two = model(tokens)
    return F.cross_entropy(one.reshape(-1, one.shape[-1]), tokens[:, 1:].reshape(-1)) + \
        F.cross_entropy(two.reshape(-1, two.shape[-1]), tokens[:, 2:].reshape(-1))


def parameter_groups(model: CharacterTransformer) -> dict[str, list[nn.Parameter]]:
    return {
        "embeddings": list(model.token_embedding.parameters()) + list(model.position_embedding.parameters()),
        **{f"block_{i}": list(block.parameters()) for i, block in enumerate(model.blocks)},
        "final_norm": list(model.final_norm.parameters()),
        "heads": list(model.head_next.parameters()) + list(model.head_two.parameters()),
    }


def relative_update_ratios(before: dict[str, list[torch.Tensor]], model: CharacterTransformer,
                           eps: float = 1e-12) -> dict[str, float]:
    ratios = {}
    for name, params in parameter_groups(model).items():
        delta_sq = sum(float(torch.sum((p.detach() - old) ** 2)) for p, old in zip(params, before[name], strict=True))
        weight_sq = sum(float(torch.sum(old ** 2)) for old in before[name])
        ratios[name] = math.sqrt(delta_sq) / max(math.sqrt(weight_sq), eps)
    return ratios


def schedule_lr(kind: str, step: int, peak_lr: float, warmup: int,
                planned_steps: int = 300, stable_through: int = 240, min_factor: float = 0.1) -> float:
    assert 1 <= step <= planned_steps and 1 <= warmup < planned_steps
    if step <= warmup:
        return peak_lr * step / warmup
    if kind == "cosine":
        progress = (step - warmup) / (planned_steps - warmup)
        factor = min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
        return peak_lr * factor
    if kind == "wsd":
        if step <= stable_through:
            return peak_lr
        progress = (step - stable_through) / (planned_steps - stable_through)
        return peak_lr * (1 - (1 - min_factor) * progress)
    if kind == "warmup_stable":
        return peak_lr
    raise ValueError(f"unknown schedule {kind}")


@dataclass(frozen=True)
class TrainConfig:
    width: int
    peak_lr: float
    warmup: int
    steps: int
    planned_steps: int
    schedule: str
    batch_size: int
    context: int
    seed: int = BASE_SEED
    weight_decay: float = 0.1
    log_ratios: bool = False


class CharacterData:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = chars
        encode = {ch: i for i, ch in enumerate(chars)}
        ids = torch.tensor([encode[ch] for ch in text], dtype=torch.long)
        split = int(0.9 * len(ids))
        self.train, self.val = ids[:split], ids[split:]

    @staticmethod
    def batches(ids: torch.Tensor, seed: int, count: int, batch_size: int, context: int) -> list[torch.Tensor]:
        assert len(ids) > context + 1
        gen = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, len(ids) - context, (count, batch_size), generator=gen)
        offsets = torch.arange(context)
        return ids[starts.unsqueeze(-1) + offsets]


def load_text(cache: Path, full: bool) -> str:
    if not full:
        base = "To be, or not to be, that is the question.\nAll the world's a stage.\n"
        return base * 300
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(DATA_URL, cache)
    payload = cache.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == DATA_SHA256
    return payload.decode("utf-8")


@torch.no_grad()
def evaluate(model: nn.Module, batches: list[torch.Tensor], device: torch.device) -> float:
    model.eval()
    losses = [float(objective(model, batch.to(device))) for batch in batches]
    model.train()
    return float(np.mean(losses))


def train_once(config: TrainConfig, data: CharacterData, device: torch.device,
               val_batches: list[torch.Tensor]) -> dict[str, Any]:
    seed_everything(config.seed)
    model = CharacterTransformer(len(data.vocab), config.width, config.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.peak_lr,
                                  betas=(0.9, 0.95), weight_decay=config.weight_decay)
    batches = CharacterData.batches(data.train, BASE_SEED + 100, config.steps,
                                    config.batch_size, config.context)
    history, ratio_history = [], []
    for step, batch in enumerate(batches, 1):
        lr = schedule_lr(config.schedule, step, config.peak_lr, config.warmup,
                         config.planned_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        before = None
        if config.log_ratios:
            before = {name: [p.detach().clone() for p in params]
                      for name, params in parameter_groups(model).items()}
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, batch.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step, "lr": lr, "train_loss": float(loss.detach())})
        if before is not None:
            ratio_history.append({"step": step, **relative_update_ratios(before, model)})
    val_loss = evaluate(model, val_batches, device)
    return {"final_train_loss": history[-1]["train_loss"], "final_val_loss": val_loss,
            "history": history, "ratio_history": ratio_history,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "parameter_count": sum(p.numel() for p in model.parameters())}


def _expected(profile: str, kind: str, width: int = 128, steps: int = 1) -> float:
    if profile == "smoke":
        return max(0.1, steps * 0.08 * (width / 32) ** 2)
    base = {"tuning": 0.055, "final": 0.065, "width": 0.060}.get(kind, 0.01)
    return max(0.5, steps * base * (width / 256) ** 2)


def _strip_state(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "state_dict"}


def tune_schedulers(profile: str, data: CharacterData, device: torch.device,
                    val_batches: list[torch.Tensor], logger: RunLogger,
                    width: int, steps: int, planned: int, batch: int, context: int) -> dict[str, Any]:
    lrs = [1.5e-4, 3e-4, 6e-4, 1.2e-3] if profile == "full" else [3e-4, 6e-4]
    warmups = [10, 20, 40] if profile == "full" else [1, 2]
    all_rows, best = {}, {}
    for scheduler in ("cosine", "wsd"):
        rows = []
        for lr in lrs:
            for warmup in warmups:
                cfg = TrainConfig(width, lr, warmup, steps, planned, scheduler, batch, context)
                run_id = f"tune-{scheduler}-lr{lr:g}-wu{warmup}"
                result = logger.run(run_id, "scheduler_tuning", asdict(cfg), cfg.seed,
                                    _expected(profile, "tuning", width, steps),
                                    lambda cfg=cfg: train_once(cfg, data, device, val_batches))
                rows.append({"lr": lr, "warmup": warmup,
                             "train_loss": result["final_train_loss"],
                             "val_loss": result["final_val_loss"]})
        all_rows[scheduler] = rows
        best[scheduler] = min(rows, key=lambda x: x["val_loss"])
    finals, states = {}, {}
    for scheduler in ("cosine", "wsd"):
        chosen = best[scheduler]
        cfg = TrainConfig(width, chosen["lr"], chosen["warmup"], steps, planned,
                          scheduler, batch, context, log_ratios=True)
        result = logger.run(f"final-{scheduler}", "scheduler_final", asdict(cfg), cfg.seed,
                            _expected(profile, "final", width, steps),
                            lambda cfg=cfg: train_once(cfg, data, device, val_batches))
        finals[scheduler] = _strip_state(result)
        states[scheduler] = result["state_dict"]
    winner = min(finals, key=lambda key: finals[key]["final_val_loss"])
    return {"candidates": all_rows, "best_configurations": best, "finals": finals,
            "winner": winner, "retained_state_dict": states[winner]}


def geometric_midpoint(a: float, b: float) -> float:
    return math.sqrt(a * b)


def width_sweep(profile: str, data: CharacterData, device: torch.device,
                val_batches: list[torch.Tensor], logger: RunLogger,
                context: int, batch: int) -> dict[str, Any]:
    widths = [256, 512, 1024] if profile == "full" else [32, 64, 96]
    initial = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3] if profile == "full" else [2e-4, 5e-4, 1e-3]
    steps, warmup, planned = ((120, 20, 120) if profile == "full" else (4, 1, 4))
    seeds = [42, 314, 2718]
    output: dict[str, Any] = {}
    per_seed_best: dict[int, dict[int, float]] = {seed: {} for seed in seeds}
    for width in widths:
        observations: dict[tuple[float, int], dict[str, float]] = {}

        def run_point(lr: float, seed: int) -> None:
            key = (lr, seed)
            if key in observations:
                return
            cfg = TrainConfig(width, lr, warmup, steps, planned, "warmup_stable", batch, context, seed)
            result = logger.run(f"width-{width}-lr{lr:.8g}-seed{seed}", "width_sweep",
                                asdict(cfg), seed, _expected(profile, "width", width, steps),
                                lambda cfg=cfg: train_once(cfg, data, device, val_batches))
            observations[key] = {"train_loss": result["final_train_loss"],
                                 "val_loss": result["final_val_loss"]}

        grid = list(initial)
        for lr in grid:
            run_point(lr, seeds[0])
        ranked = sorted(grid)
        best_lr = min(ranked, key=lambda lr: observations[(lr, seeds[0])]["val_loss"])
        extensions = 0
        while best_lr in (ranked[0], ranked[-1]) and extensions < 2:
            new_lr = ranked[0] / 2 if best_lr == ranked[0] else ranked[-1] * 2
            ranked.append(new_lr)
            ranked.sort()
            run_point(new_lr, seeds[0])
            best_lr = min(ranked, key=lambda lr: observations[(lr, seeds[0])]["val_loss"])
            extensions += 1
        idx = ranked.index(best_lr)
        if 0 < idx < len(ranked) - 1:
            for lr in (geometric_midpoint(ranked[idx - 1], best_lr),
                       geometric_midpoint(best_lr, ranked[idx + 1])):
                ranked.append(lr)
                run_point(lr, seeds[0])
            ranked.sort()
            best_lr = min(ranked, key=lambda lr: observations[(lr, seeds[0])]["val_loss"])
        idx = ranked.index(best_lr)
        neighbours = ranked[max(0, idx - 1):idx + 2]
        if len(neighbours) < 3:
            # A persistent boundary minimum is low confidence; still repeat it and two nearest tested points.
            neighbours = ranked[:3] if idx == 0 else ranked[-3:]
        for lr in neighbours:
            for seed in seeds:
                run_point(lr, seed)
        repeated = set(neighbours)
        selected = best_lr
        # If the three-seed mean moves the minimum to an edge of the repeated
        # neighbourhood, repeat that point's newly exposed neighbour too.
        for _ in range(len(ranked)):
            means = {lr: float(np.mean([observations[(lr, seed)]["val_loss"] for seed in seeds]))
                     for lr in repeated}
            new_selected = min(means, key=means.get)
            idx = ranked.index(new_selected)
            required = set(ranked[max(0, idx - 1):idx + 2])
            missing = required - repeated
            selected = new_selected
            if not missing:
                break
            for lr in missing:
                for seed in seeds:
                    run_point(lr, seed)
            repeated |= missing
        neighbours = sorted(repeated)
        for seed in seeds:
            per_seed_best[seed][width] = min(neighbours, key=lambda lr: observations[(lr, seed)]["val_loss"])
        output[str(width)] = {
            "initial_grid": initial, "tested_grid": ranked, "neighbours_repeated": neighbours,
            "empirical_best_lr": selected, "boundary_minimum": selected in (ranked[0], ranked[-1]),
            "observations": [{"lr": lr, "seed": seed, **values}
                             for (lr, seed), values in sorted(observations.items())],
            "mean_val_loss_repeated": means,
        }
    x = np.log(np.array(widths, dtype=float))
    y = np.log(np.array([output[str(w)]["empirical_best_lr"] for w in widths]))
    exponent, intercept = np.polyfit(x, y, 1)
    fitted = intercept + exponent * x
    ss_res, ss_tot = float(np.sum((y - fitted) ** 2)), float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 and ss_res == 0 else 1 - ss_res / max(ss_tot, 1e-30)
    prediction = float(math.exp(intercept) * 4096**exponent)
    seed_predictions = []
    for seed in seeds:
        sy = np.log([per_seed_best[seed][w] for w in widths])
        se, si = np.polyfit(x, sy, 1)
        seed_predictions.append(float(math.exp(si) * 4096**se))
    prediction_range = [min(seed_predictions), max(seed_predictions)]
    ratio = prediction_range[1] / max(prediction_range[0], 1e-30)
    interior = all(not output[str(w)]["boundary_minimum"] for w in widths)
    stable = all(sum(math.isclose(per_seed_best[s][w], output[str(w)]["empirical_best_lr"])
                     for s in seeds) >= 2 for w in widths)
    confidence = "high" if interior and stable and r2 >= 0.9 and ratio <= 2 else \
        "moderate" if interior and r2 >= 0.6 and ratio <= 4 else "low"
    return {"widths": output, "fit": {"exponent": float(exponent), "intercept": float(intercept),
            "r_squared": r2, "predicted_lr_width_4096": prediction,
            "seed_predictions": seed_predictions, "prediction_range": prediction_range,
            "prediction_range_ratio": ratio, "interior_minima": interior,
            "stable_minima": stable, "confidence": confidence,
            "confidence_rule": "high: interior/stable minima, R^2>=0.9, range<=2x; moderate: interior minima, R^2>=0.6, range<=4x; low otherwise"}}


def make_plots(metrics: dict[str, Any], out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/optimizer-mpl")
    import matplotlib.pyplot as plt
    diag = metrics["adam_bias_correction"]["first_20"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (("corrected_weight", "bias corrected"), ("uncorrected_weight", "uncorrected")):
        axes[0].plot([r["step"] for r in diag], [r[key] for r in diag], marker="o", label=label)
    for key, label in (("corrected_update", "bias corrected"), ("uncorrected_update", "uncorrected")):
        axes[1].plot([r["step"] for r in diag], [abs(r[key]) for r in diag], marker="o", label=label)
    axes[0].set(title="Parameter trajectory", xlabel="step", ylabel="weight")
    axes[1].set(title="Update magnitude", xlabel="step", ylabel="|update|")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "adam_bias_correction.png", dpi=160); plt.close(fig)

    for scheduler, final in metrics["scheduler_comparison"]["finals"].items():
        rows = final["ratio_history"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for layer in [k for k in rows[0] if k != "step"]:
            ax.plot([r["step"] for r in rows], [r[layer] for r in rows], label=layer)
        warmup = metrics["scheduler_comparison"]["best_configurations"][scheduler]["warmup"]
        ax.axvline(warmup, color="black", linestyle="--", label=f"warmup endpoint ({warmup})")
        ax.set(xlabel="optimizer step", ylabel="||delta W||2 / max(||W||2, eps)",
               title=f"Layer-relative updates: {scheduler}")
        ax.set_yscale("log"); ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / f"relative_updates_{scheduler}.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for width, info in metrics["width_sweep"]["widths"].items():
        grouped: dict[float, list[float]] = {}
        for row in info["observations"]:
            grouped.setdefault(row["lr"], []).append(row["val_loss"])
        xs = sorted(grouped)
        ys = [float(np.mean(grouped[x])) for x in xs]
        ax.plot(xs, ys, marker="o", label=f"width {width}")
        best = info["empirical_best_lr"]
        ax.scatter([best], [float(np.mean(grouped[best]))], marker="*", s=180)
    ax.set_xscale("log"); ax.set(xlabel="learning rate", ylabel="held-out loss",
                                 title="Width learning-rate sweep")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "width_lr_sweep.png", dpi=160); plt.close(fig)


def validate_metrics(metrics: dict[str, Any]) -> None:
    assert metrics["provenance"]["result_kind"] == "colab_t4_full"
    assert metrics["provenance"]["device"] and "T4" in metrics["provenance"]["device"].upper()
    assert len(metrics["adam_manual"]["rows"]) == 5
    assert metrics["adam_manual"]["pytorch_float64_parity"] is True
    assert len(metrics["adam_bias_correction"]["first_20"]) == 20
    assert metrics["adam_bias_correction"]["stops_mattering_step"] > 20
    assert metrics["scheduler_comparison"]["planned_steps"] == 300
    assert metrics["scheduler_comparison"]["stopped_at_step"] == 200
    assert set(metrics["width_sweep"]["widths"]) == {"256", "512", "1024"}
    assert all(r["status"] == "ok" for r in metrics["runs"])
    assert all(RunLogger.REQUIRED <= r.keys() for r in metrics["runs"])
    expected_layers = {"step", "embeddings", "block_0", "block_1", "block_2",
                       "block_3", "final_norm", "heads"}
    for scheduler in ("cosine", "wsd"):
        assert len(metrics["scheduler_comparison"]["candidates"][scheduler]) == 12
        final = metrics["scheduler_comparison"]["finals"][scheduler]
        assert len(final["history"]) == 200
        assert len(final["ratio_history"]) == 200
        assert all(set(row) == expected_layers for row in final["ratio_history"])
        assert [row["step"] for row in final["ratio_history"]] == list(range(1, 201))


def generate_readme(metrics_path: Path, output_path: Path) -> str:
    m = json.loads(metrics_path.read_text())
    validate_metrics(m)
    adam = m["adam_manual"]
    sched = m["scheduler_comparison"]
    sweep = m["width_sweep"]
    fit = sweep["fit"]
    rows = adam["rows"]
    table = "\n".join(f"| {r['step']} | {r['gradient']:.3f} | {r['m']:.9f} | {r['v']:.9f} | {r['m_hat']:.9f} | {r['v_hat']:.9f} | {r['update']:.9f} | {r['new_weight']:.9f} |" for r in rows)
    tuning_lines = []
    for name in ("cosine", "wsd"):
        b, f = sched["best_configurations"][name], sched["finals"][name]
        tuning_lines.append(f"| {name.upper()} | {b['lr']:.3g} | {b['warmup']} | {f['final_train_loss']:.6f} | {f['final_val_loss']:.6f} |")
    width_lines = []
    for width, info in sweep["widths"].items():
        vals = [r["val_loss"] for r in info["observations"] if math.isclose(r["lr"], info["empirical_best_lr"])]
        width_lines.append(f"| {width} | {info['empirical_best_lr']:.4g} | {np.mean(vals):.6f} | {np.std(vals):.6f} | {'yes' if info['boundary_minimum'] else 'no'} |")
    timing_lines = []
    for r in m["runs"]:
        cfg = r["configuration"]
        compact = ", ".join(f"{k}={v}" for k, v in cfg.items()
                            if k in {"schedule", "width", "peak_lr", "warmup", "steps", "profile"}) or "-"
        timing_lines.append(f"| {r['run_id']} | {r['experiment']} | {compact} | {r['expected_seconds']:.2f} | {r['duration_seconds']:.2f} | {r['status']} | {r['peak_gpu_memory_bytes'] / 2**20:.1f} |")
    components = [r for r in m["runs"] if r["experiment"] != "total_runtime"]
    measured_total = sum(r["duration_seconds"] for r in components)
    expected_total = sum(r["expected_seconds"] for r in components)
    text = f"""# Transformer optimizer benchmark

[Executed Colab notebook](transformer_optimizer_benchmark_colab.ipynb) · [machine-readable metrics](results/metrics.json) · [run log](results/run_log.csv)

All numerical claims below were generated strictly from `results/metrics.json`, after the full internal assertion suite passed. The run used **{m['provenance']['device']}**, fp32, fixed seeds, fixed validation batches, identical candidate initialization/data order, and Tiny Shakespeare with its pinned SHA-256.

## 1. Adam by hand and against PyTorch

For the first gradient, `m = 0.9(0) + 0.1({rows[0]['gradient']}) = {rows[0]['m']:.9f}` and `v = 0.999(0) + 0.001({rows[0]['gradient']}^2) = {rows[0]['v']:.9f}`. Bias correction gives `m_hat = m/(1-0.9^1) = {rows[0]['m_hat']:.9f}` and `v_hat = v/(1-0.999^1) = {rows[0]['v_hat']:.9f}`. Thus `update = 0.01*m_hat/(sqrt(v_hat)+1e-8) = {rows[0]['update']:.9f}` and the new weight is `{rows[0]['new_weight']:.9f}`.

| Step | Gradient | m | v | m-hat | v-hat | Update | New weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

Every value was independently recovered from PyTorch's float64 Adam state and asserted equal to tight floating-point tolerance. Under cyclic repetition of these gradients, bias correction “stops mattering” at step **{m['adam_bias_correction']['stops_mattering_step']}** by the declared rule: {m['adam_bias_correction']['definition']}.

![Adam bias correction](results/adam_bias_correction.png)

## 2. Cosine versus WSD at step 200

Both schedules were configured for 300 steps and deliberately stopped at step 200. Cosine would decay to 10% of peak at step 300. WSD warms up, stays at peak through step 240, and would decay over steps 241–300. Each received the same 12-candidate tuning budget.

| Schedule | Tuned peak LR | Warmup | Step-200 train loss | Held-out loss |
|---|---:|---:|---:|---:|
{chr(10).join(tuning_lines)}

The retained model is **{sched['winner'].upper()}**, selected only by lower held-out loss. The vertical marker in each ratio plot is the tuned warmup endpoint: the point where warmup stops directly scaling the learning rate. It does not claim that model-driven update-ratio changes cease.

![Cosine relative updates](results/relative_updates_cosine.png)
![WSD relative updates](results/relative_updates_wsd.png)

## 3. Width versus best learning rate

The warmup-stable sweep began at `[1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]`; boundary minima extended the grid, while interior minima received geometric-midpoint refinement. Each selected minimum and its tested neighbours was repeated over seeds 42, 314, and 2718.

| Width | Empirical best LR | Mean held-out loss | Seed std. dev. | Boundary minimum |
|---:|---:|---:|---:|:---:|
{chr(10).join(width_lines)}

The log-log fit is `best_lr = exp({fit['intercept']:.6f}) * width^({fit['exponent']:.6f})`, with **R²={fit['r_squared']:.4f}**. It predicts **{fit['predicted_lr_width_4096']:.4g}** at width 4096; seed-wise fits span **[{fit['prediction_range'][0]:.4g}, {fit['prediction_range'][1]:.4g}]** ({fit['prediction_range_ratio']:.2f}x). Confidence is **{fit['confidence']}**, using the predeclared rule: {fit['confidence_rule']}.

![Width sweep](results/width_lr_sweep.png)

## 4. Timing and audit trail

CUDA was synchronized immediately before and after every timed run. The log records each run ID, experiment, full configuration, seed, expected and measured duration, final losses, status, and peak allocated GPU memory. Expected timed work totaled **{readable_duration(expected_total)}**; measured timed work totaled **{readable_duration(measured_total)}**; notebook wall time was **{m['timing']['total_readable']}**. Setup, diagnostics, plotting, packaging, every candidate, both finals, and every width/LR point appear individually in the JSON/CSV logs.

## Reproducibility

- Source commit at execution: `{m['provenance']['source_commit']}`
- UTC start: `{m['provenance']['started_at']}`
- Python `{m['provenance']['python']}`; PyTorch `{m['provenance']['torch']}`
- Dataset SHA-256: `{DATA_SHA256}`
- Model: four layers, four heads, two untied character-prediction heads, context {m['provenance']['context']}
- Scheduler comparisons use width {m['provenance']['comparison_width']} and stop exactly at update 200.

### Per-run expected and measured durations

| Run ID | Experiment | Configuration | Expected seconds | Measured seconds | Status | Peak GPU MiB |
|---|---|---|---:|---:|---|---:|
{chr(10).join(timing_lines)}

The full raw evidence is under [`results/`](results/). Local smoke outputs are intentionally written elsewhere and cannot generate this README.
"""
    output_path.write_text(text)
    return text


def run_pipeline(profile: str, out_dir: Path) -> dict[str, Any]:
    full = profile == "full"
    if full:
        assert torch.cuda.is_available(), "full results require a CUDA runtime"
        assert "T4" in torch.cuda.get_device_name(0).upper(), "full results require a free Colab T4"
        assert out_dir.name == "results", "full results must be written under results/"
    else:
        assert out_dir.name != "results", "smoke results must never overwrite submission results"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger()
    cuda_sync()
    wall_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    device = torch.device("cuda" if full else "cpu")

    def setup() -> dict[str, Any]:
        text = load_text(Path(".cache/tinyshakespeare.txt"), full)
        return {"data": CharacterData(text), "final_train_loss": None, "final_val_loss": None}
    setup_result = logger.run("setup", "setup", {"profile": profile}, BASE_SEED, 20 if full else .1, setup)
    data: CharacterData = setup_result["data"]
    context, batch = ((128, 8) if full else (16, 2))
    val_batches = CharacterData.batches(data.val, BASE_SEED + 1, 4, batch, context)

    def diagnostics() -> dict[str, Any]:
        rows, ref = adam_manual(), adam_torch_reference()
        assert_adam_parity(rows, ref)
        return {"rows": rows, "bias": bias_correction_diagnostic(),
                "final_train_loss": None, "final_val_loss": None}
    diag = logger.run("optimizer-diagnostics", "optimizer_diagnostics", {}, None, 1, diagnostics)
    width, steps, planned = ((256, 200, 300) if full else (32, 4, 6))
    sched = tune_schedulers(profile, data, device, val_batches, logger, width, steps, planned, batch, context)
    sweep = width_sweep(profile, data, device, val_batches, logger, context, batch)
    retained = sched.pop("retained_state_dict")
    if full:
        torch.save(retained, out_dir / "retained_model.pt")

    metrics: dict[str, Any] = {
        "provenance": {"result_kind": "colab_t4_full" if full else "local_cpu_smoke",
                       "device": torch.cuda.get_device_name(0) if full else "CPU smoke (synthetic data)",
                       "started_at": started_at, "python": platform.python_version(),
                       "torch": torch.__version__, "source_commit": os.environ.get("SOURCE_COMMIT", "uncommitted"),
                       "context": context, "comparison_width": width},
        "adam_manual": {"initial_weight": 1.0, "lr": .01, "beta1": .9, "beta2": .999,
                        "epsilon": 1e-8, "gradients": ADAM_GRADIENTS, "rows": diag["rows"],
                        "pytorch_float64_parity": True},
        "adam_bias_correction": diag["bias"],
        "scheduler_comparison": {**sched, "planned_steps": planned, "stopped_at_step": steps,
                                 "cosine_min_factor": .1, "wsd_stable_through": 240 if full else 4},
        "width_sweep": sweep,
    }
    logger.run("plotting", "plotting", {}, None, 5 if full else 1,
               lambda: (make_plots(metrics, out_dir) or {"final_train_loss": None, "final_val_loss": None}))
    metrics["runs"] = logger.records
    metrics_path = out_dir / "metrics.json"

    def package() -> dict[str, Any]:
        metrics["runs"] = logger.records
        metrics["timing"] = {"total_seconds": time.perf_counter() - wall_start,
                             "total_readable": readable_duration(time.perf_counter() - wall_start)}
        metrics_path.write_text(json.dumps(metrics, indent=2))
        if full:
            validate_metrics(metrics)
        # Packaging is gated on all expected plot and log inputs being present.
        assert all((out_dir / name).exists() for name in
                   ("adam_bias_correction.png", "relative_updates_cosine.png",
                    "relative_updates_wsd.png", "width_lr_sweep.png"))
        return {"final_train_loss": None, "final_val_loss": None}

    logger.run("packaging", "packaging", {"profile": profile}, None, 2 if full else .2, package)
    cuda_sync()
    total = time.perf_counter() - wall_start
    total_expected = sum(r["expected_seconds"] for r in logger.records)
    logger.records.append({"run_id": "total-runtime", "experiment": "total_runtime",
                           "configuration": {"profile": profile}, "seed": None,
                           "duration_seconds": total, "readable_duration": readable_duration(total),
                           "final_train_loss": None, "final_val_loss": None, "status": "ok",
                           "peak_gpu_memory_bytes": max((r["peak_gpu_memory_bytes"] for r in logger.records), default=0),
                           "expected_seconds": total_expected})
    metrics["runs"] = logger.records
    metrics["timing"] = {"total_seconds": total, "total_readable": readable_duration(total),
                         "cuda_synchronized": True}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.write(out_dir)
    if full:
        validate_metrics(metrics)
    return metrics


def run_full() -> dict[str, Any]:
    return run_pipeline("full", Path("results"))


def run_smoke() -> dict[str, Any]:
    return run_pipeline("smoke", Path("smoke_results"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    metrics = run_full() if args.profile == "full" else run_smoke()
    print(json.dumps({"kind": metrics["provenance"]["result_kind"],
                      "duration": metrics["timing"]["total_readable"]}, indent=2))


if __name__ == "__main__":
    main()
