# Optimizer truth: Adam, schedules, and width scaling

This repository contains a Colab-first optimizer assignment for a four-layer,
four-head nanoGPT-style character Transformer. The full run compares Adam bias
correction, cosine decay, WSD, and width-dependent learning-rate choices under
controlled initialization, batches, validation data, and timing.

The measured report is intentionally not pre-populated. A successful T4 run writes
`results/metrics.json`, CSV/JSON logs, four PNG plots, the retained checkpoint, and
then regenerates this README strictly from those metrics. Local smoke measurements
cannot pass the README-generation gate.

## Files

- [`optimizer_truth_colab.ipynb`](optimizer_truth_colab.ipynb): Colab entry point,
  artifact capture, assertions, and optional push-back workflow.
- [`optimizer_experiments.py`](optimizer_experiments.py): model, diagnostics,
  schedulers, tuning, adaptive width sweep, logging, plotting, and report generation.
- [`tests/`](tests): local arithmetic, determinism, scheduler, aggregation, timing,
  and report-gating tests.

## Local verification

```bash
python -m pytest -q
MPLCONFIGDIR=/tmp/optimizer-mpl python optimizer_experiments.py --profile smoke
```

The smoke run uses synthetic text on CPU and writes only to ignored
`smoke_results/`. It covers the complete orchestration but is not assignment evidence.

## Colab workflow

1. Push this implementation on branch `colab-results` and open
   `optimizer_truth_colab.ipynb` in a free Colab T4 runtime.
2. Set `REPO_URL` in the first code cell and add `GITHUB_TOKEN` to Colab Secrets
   with repository write access. The token is read into process memory only and is
   never printed, written to disk, embedded in the remote URL, or committed.
3. Run all cells. The full pipeline is guarded to accept only a CUDA device whose
   name contains `T4`.
4. Inspect the final assertion summary, measured timing table, plots, and losses.
   The last cell captures the executed notebook, commits generated evidence, and
   pushes it to `colab-results` when `PUSH_RESULTS=True`.

Expected wall time is approximately 45–90 minutes. Every timed component and total
runtime synchronizes CUDA at its boundaries. After the Colab results return, they
must be reviewed before the generated README or retained model is accepted.
