import json
import math
from pathlib import Path

import torch

import optimizer_experiments as oe


def test_adam_manual_matches_pytorch_float64():
    manual = oe.adam_manual()
    reference = oe.adam_torch_reference()
    oe.assert_adam_parity(manual, reference)
    assert len(manual) == 5
    assert manual[0]["new_weight"] < 1.0


def test_bias_correction_definition_is_consecutive():
    result = oe.bias_correction_diagnostic()
    assert len(result["first_20"]) == 20
    assert result["stops_mattering_step"] > 1


def test_scheduler_boundaries():
    peak = 1e-3
    assert oe.schedule_lr("cosine", 10, peak, 10) == peak
    assert math.isclose(oe.schedule_lr("cosine", 300, peak, 10), peak * 0.1)
    assert oe.schedule_lr("wsd", 20, peak, 20) == peak
    assert oe.schedule_lr("wsd", 240, peak, 20) == peak
    assert oe.schedule_lr("wsd", 241, peak, 20) < peak
    assert math.isclose(oe.schedule_lr("wsd", 300, peak, 20), peak * 0.1)


def test_ratio_aggregation_uses_group_l2_norms():
    model = oe.CharacterTransformer(vocab_size=7, width=8, context=4, layers=4, heads=4)
    before = {name: [p.detach().clone() for p in params]
              for name, params in oe.parameter_groups(model).items()}
    with torch.no_grad():
        model.head_next.weight.add_(0.25)
    ratios = oe.relative_update_ratios(before, model)
    assert ratios["heads"] > 0
    assert all(ratios[name] == 0 for name in ratios if name != "heads")


def test_batches_are_deterministic_and_seed_sensitive():
    ids = torch.arange(200)
    a = oe.CharacterData.batches(ids, 7, 3, 2, 8)
    b = oe.CharacterData.batches(ids, 7, 3, 2, 8)
    c = oe.CharacterData.batches(ids, 8, 3, 2, 8)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert any(not torch.equal(x, y) for x, y in zip(a, c))


def test_timing_record_schema(tmp_path):
    logger = oe.RunLogger()
    result = logger.run("x", "unit", {"a": 1}, 42, 0.1,
                        lambda: {"final_train_loss": 2.0, "final_val_loss": 3.0})
    assert result["final_val_loss"] == 3.0
    assert oe.RunLogger.REQUIRED <= logger.records[0].keys()
    logger.write(tmp_path)
    assert json.loads((tmp_path / "run_log.json").read_text())[0]["status"] == "ok"


def _full_metrics_fixture():
    first = oe.adam_manual()
    ratio = [{"step": 1, "embeddings": .1, "block_0": .1, "block_1": .1,
              "block_2": .1, "block_3": .1, "final_norm": .1, "heads": .1}]
    candidates = [{"lr": 3e-4, "warmup": 20, "train_loss": 5.0, "val_loss": 5.1}] * 12
    width = {str(w): {"empirical_best_lr": 3e-4, "boundary_minimum": False,
                      "observations": [{"lr": 3e-4, "seed": s, "val_loss": 5.0 + s * 0,
                                        "train_loss": 4.9} for s in (42, 314, 2718)]}
             for w in (256, 512, 1024)}
    run = {"run_id": "x", "experiment": "unit", "configuration": {}, "seed": 42,
           "duration_seconds": 1.0, "readable_duration": "1.00s", "final_train_loss": 5.0,
           "final_val_loss": 5.1, "status": "ok", "peak_gpu_memory_bytes": 0,
           "expected_seconds": 2.0}
    return {"provenance": {"result_kind": "colab_t4_full", "device": "Tesla T4",
                            "source_commit": "abc", "started_at": "now", "python": "3.12",
                            "torch": "2.x", "context": 128, "comparison_width": 256},
            "adam_manual": {"rows": first},
            "adam_bias_correction": {"stops_mattering_step": 100,
                                      "definition": "test definition"},
            "scheduler_comparison": {"planned_steps": 300, "stopped_at_step": 200,
                "candidates": {"cosine": candidates, "wsd": candidates},
                "best_configurations": {"cosine": {"lr": 3e-4, "warmup": 20},
                                        "wsd": {"lr": 3e-4, "warmup": 20}},
                "finals": {"cosine": {"final_train_loss": 5.0, "final_val_loss": 5.1,
                                        "ratio_history": ratio},
                           "wsd": {"final_train_loss": 5.0, "final_val_loss": 5.2,
                                   "ratio_history": ratio}}, "winner": "cosine"},
            "width_sweep": {"widths": width, "fit": {"exponent": -0.5, "intercept": -5,
                "r_squared": .95, "predicted_lr_width_4096": 1e-4,
                "prediction_range": [8e-5, 1.2e-4], "prediction_range_ratio": 1.5,
                "confidence": "high", "confidence_rule": "fixture"}},
            "runs": [run], "timing": {"total_readable": "1m 0s"}}


def test_readme_generation_is_metrics_gated(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_full_metrics_fixture()))
    output = tmp_path / "README.md"
    text = oe.generate_readme(metrics, output)
    assert text == output.read_text()
    assert "Tesla T4" in text and "R²=0.9500" in text
    bad = _full_metrics_fixture()
    bad["provenance"]["result_kind"] = "local_cpu_smoke"
    metrics.write_text(json.dumps(bad))
    try:
        oe.generate_readme(metrics, output)
    except AssertionError:
        pass
    else:
        raise AssertionError("smoke metrics must not generate the submission README")
