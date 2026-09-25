import json
import math
from pathlib import Path

import torch

import additional_experiments as additional
import transformer_optimizer_benchmark as oe


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
    ratio = [{"step": step, "embeddings": .1, "block_0": .1, "block_1": .1,
              "block_2": .1, "block_3": .1, "final_norm": .1, "heads": .1}
             for step in range(1, 201)]
    history = [{"step": step, "lr": 3e-4, "train_loss": 5.0}
               for step in range(1, 201)]
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
            "adam_manual": {"rows": first, "pytorch_float64_parity": True},
            "adam_bias_correction": {"first_20": [{}] * 20,
                                      "stops_mattering_step": 100,
                                      "definition": "test definition"},
            "scheduler_comparison": {"planned_steps": 300, "stopped_at_step": 200,
                "candidates": {"cosine": candidates, "wsd": candidates},
                "best_configurations": {"cosine": {"lr": 3e-4, "warmup": 20},
                                        "wsd": {"lr": 3e-4, "warmup": 20}},
                "finals": {"cosine": {"final_train_loss": 5.0, "final_val_loss": 5.1,
                                        "history": history,
                                        "ratio_history": ratio},
                           "wsd": {"final_train_loss": 5.0, "final_val_loss": 5.2,
                                   "history": history,
                                   "ratio_history": ratio}}, "winner": "cosine"},
            "width_sweep": {"widths": width, "fit": {"exponent": -0.5, "intercept": -5,
                "r_squared": .95, "predicted_lr_width_4096": 1e-4,
                "prediction_range": [8e-5, 1.2e-4], "prediction_range_ratio": 1.5,
                "confidence": "high", "confidence_rule": "fixture"}},
            "runs": [run], "timing": {"total_readable": "1m 0s"}}


def test_metrics_validation_rejects_smoke_results(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_full_metrics_fixture()))
    oe.validate_metrics(json.loads(metrics.read_text()))
    bad = _full_metrics_fixture()
    bad["provenance"]["result_kind"] = "local_cpu_smoke"
    try:
        oe.validate_metrics(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("smoke metrics must not pass full-result validation")


def test_additional_plan_expands_and_replicates_search():
    plan = additional.additional_plan()
    assert plan["seeds"] == [42, 314, 2718]
    assert max(plan["scheduler_lrs"]) > 1.2e-3
    assert min(plan["warmups"]) < 10 < max(plan["warmups"])
    assert set(plan["width_grids"]) == {2048, 4096}
    assert 2.5e-5 in plan["width_grids"][4096]
    assert plan["validation_batches"] == 32


def test_additional_parts_cover_full_work_without_overlap():
    assert additional.ADDITIONAL_PARTS[0:2] == ("scheduler", "width_2048")
    assert len(additional.ADDITIONAL_PARTS) == 7
    assert additional.part_width_grids("scheduler") == {}
    assert additional.part_width_grids("width_2048") == {
        2048: additional.FULL_WIDTH_GRIDS[2048]
    }
    sharded_lrs = [
        additional.part_width_grids(part)[4096][0]
        for part in additional.ADDITIONAL_PARTS[2:]
    ]
    assert sharded_lrs == additional.FULL_WIDTH_GRIDS[4096]
    assert len(set(sharded_lrs)) == len(sharded_lrs)


def test_width_shards_merge_and_recompute_global_minimum():
    grid = [1e-5, 2e-5]
    seeds = [42, 314]
    capability = {"parameter_count": 10}
    infos = []
    for lr, loss in zip(grid, [2.0, 1.0], strict=True):
        infos.append({
            "status": "complete",
            "capability": capability,
            "observations": [
                {"lr": lr, "seed": seed, "data_seed": 100_000 + seed,
                 "train_loss": loss - 0.1, "val_loss": loss}
                for seed in seeds
            ],
        })
    merged = additional.merge_width_part_results(4096, infos, grid, seeds)
    assert merged["status"] == "complete"
    assert len(merged["observations"]) == 4
    assert merged["best"]["lr"] == 2e-5
    assert merged["boundary_minimum"] is True


def test_width_shard_merge_records_missing_observations():
    info = {
        "status": "resource_limited", "reason": "out of memory",
        "capability": {"parameter_count": 10}, "observations": [],
    }
    merged = additional.merge_width_part_results(4096, [info], [1e-5], [42, 314])
    assert merged["status"] == "resource_limited"
    assert "2 observations missing" in merged["reason"]


def test_large_width_memory_estimate_matches_architecture():
    assert additional.estimated_parameter_count(65, 256, 128) == 3_233_024
    count = additional.estimated_parameter_count(65, 4096, 128)
    assert count == 806_703_104
    assert additional.training_memory_estimate(count) == 16 * count


def test_accumulation_preserves_effective_batch_update():
    data = oe.CharacterData("abcdefghij" * 40)
    val = oe.CharacterData.batches(data.val, 99, 2, 2, 4)
    common = dict(width=8, peak_lr=3e-4, warmup=1, steps=2,
                  planned_steps=2, schedule="warmup_stable",
                  effective_batch_size=2, context=4, model_seed=7,
                  data_seed=11, capture_state=True)
    full = additional.train_controlled(
        additional.AdditionalTrainConfig(**common, micro_batch_size=2),
        data, torch.device("cpu"), val,
    )
    accumulated = additional.train_controlled(
        additional.AdditionalTrainConfig(**common, micro_batch_size=1),
        data, torch.device("cpu"), val,
    )
    assert math.isclose(full["final_train_loss"], accumulated["final_train_loss"],
                        rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(full["final_val_loss"], accumulated["final_val_loss"],
                        rel_tol=1e-6, abs_tol=1e-6)
    for name, tensor in full["state_dict"].items():
        assert torch.allclose(tensor, accumulated["state_dict"][name],
                              rtol=1e-5, atol=1e-6)
