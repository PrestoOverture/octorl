#!/usr/bin/env python3
"""Preregistered R6 chooser replay with strictly separated G1 and G2 phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from src.curriculum.failure_driven import CELLS, PI0
from src.curriculum.model_chooser import (
    aggregate_samples, cyclic_relabellings, parse_completion, query_seed, render_prompt, tv_distance,
)

SEEDS = (42, 137, 2718)
ARMS = ("fixed", "failure_driven")
STAGES = tuple(range(1, 7))
QUERY_FORMULA = "36 * (base: 8+8+2*8; paired U20: 8+8; paired U80: 8+8) = 2304"


def lora_request_id(adapter_path: str) -> int:
    """Derive a stable positive ID that fits vLLM's signed int32 mapping."""
    return int(hashlib.sha256(adapter_path.encode()).hexdigest()[:8], 16) % (2**31 - 1) + 1


def canonical_metric(value: float) -> float:
    """Remove platform-specific tail bits before serializing replay metrics."""
    return float(format(value, ".14g"))


class Backend(Protocol):
    def generate(self, prompts: Sequence[str], seeds: Sequence[int],
                 adapter_path: str | None) -> list[str]: ...


class MockBackend:
    def __init__(self, mode: str = "proportional") -> None:
        if mode not in {"constant", "proportional", "garbage"}:
            raise ValueError(f"unknown mock mode: {mode}")
        self.mode = mode

    @staticmethod
    def _rates(prompt: str) -> list[float]:
        rates = []
        for cell in CELLS:
            match = re.search(rf"^\| {re.escape(cell)} \| \d+ \| \d+ \| ([0-9]+\.[0-9]{{3}}) \|$",
                              prompt, flags=re.MULTILINE)
            if match is None:
                raise ValueError(f"prompt lacks stats row for {cell}")
            rates.append(float(match.group(1)))
        return rates

    def generate(self, prompts: Sequence[str], seeds: Sequence[int],
                 adapter_path: str | None) -> list[str]:
        del adapter_path
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have equal length")
        outputs = []
        for prompt, seed in zip(prompts, seeds, strict=True):
            if self.mode == "garbage":
                outputs.append("not JSON")
                continue
            if self.mode == "constant":
                values = [40, 30, 30]
            else:
                rates = self._rates(prompt)
                values = [max(0, int(round(rate * 100)) + ((seed >> (3 * index)) % 3) - 1)
                          for index, rate in enumerate(rates)]
                if not any(values):
                    values = [1, 1, 1]
            outputs.append(json.dumps(dict(zip(CELLS, values)), separators=(",", ":")))
        return outputs


class VllmBackend:
    """Lazy vLLM backend with the R4r evaluation engine settings."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model_path, trust_remote_code=True, max_model_len=8192,
                       gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)

    def generate(self, prompts: Sequence[str], seeds: Sequence[int],
                 adapter_path: str | None) -> list[str]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        rendered = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) for prompt in prompts]
        params = [SamplingParams(seed=int(seed), temperature=0.7, top_p=0.95, max_tokens=256)
                  for seed in seeds]
        request = None
        if adapter_path is not None:
            request = LoRARequest(Path(adapter_path).name, lora_request_id(adapter_path), adapter_path)
        outputs = self.llm.generate(rendered, sampling_params=params,
                                    lora_request=request, use_tqdm=False)
        return [item.outputs[0].text for item in outputs]


def _adapter(kind: str, seed: int, path: Path) -> dict[str, str]:
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"adapter weights are unresolved: {weights}")
    return {"id": f"{kind}_{seed}", "kind": kind, "adapter_path": str(path.resolve()),
            "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}


def default_choosers(workspace: Path) -> dict[int, list[dict[str, str]]]:
    checkpoints = workspace / "artifacts/self_improve/r5/checkpoints"
    result = {}
    for seed in SEEDS:
        u20 = checkpoints / (f"r3c_{seed}_u20" if seed in (42, 137) else "r4r_warmup_2718_u20")
        u80 = checkpoints / f"r4r_failure_driven_{seed}_u80"
        result[seed] = [_adapter("u20", seed, u20), _adapter("u80", seed, u80)]
    return result


def load_adapter_map(path: Path) -> dict[int, list[dict[str, str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {str(seed) for seed in SEEDS}:
        raise ValueError(f"adapter map must have exactly seed keys {SEEDS}")
    result = {}
    for seed in SEEDS:
        mapping = raw[str(seed)]
        if set(mapping) != {"u20", "u80"}:
            raise ValueError(f"adapter map seed {seed} must have exactly u20 and u80")
        result[seed] = [_adapter(kind, seed, Path(mapping[kind])) for kind in ("u20", "u80")]
    return result


def load_windows(workspace: Path) -> list[dict[str, Any]]:
    root = workspace / "artifacts/self_improve/r4r/remote_records"
    windows = []
    for seed in SEEDS:
        for arm in ARMS:
            for stage in STAGES:
                path = root / f"seed_{seed}/{arm}/stage_{stage}/stage_distribution.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                stats = {cell: {key: int(data["cells"][cell][key]) for key in ("n", "fail", "infra")}
                         for cell in CELLS}
                windows.append({"seed": seed, "arm": arm, "stage": stage, "stats": stats,
                                "source": str(path.relative_to(workspace))})
    if len(windows) != 36:
        raise AssertionError(f"expected 36 replay windows, got {len(windows)}")
    return windows


def query_specs(workspace: Path, choosers: Mapping[int, Sequence[Mapping[str, str]]],
                *, smoke: bool = False) -> list[dict[str, Any]]:
    specs = []
    windows = load_windows(workspace)
    if smoke:
        windows = [window for window in windows
                   if (window["seed"], window["arm"], window["stage"]) == (137, "failure_driven", 1)]
    for window_index, window in enumerate(windows):
        variants = [("sample", 0, window["stats"])]
        if not smoke:
            variants += [("noise", 0, window["stats"])]
            variants += [(f"relabel_{index}", index, stats)
                         for index, stats in enumerate(cyclic_relabellings(window["stats"]), 1)]
        for query_kind, relabel_index, stats in variants:
            rendered = render_prompt(
                stats, prereg_path=workspace / "artifacts/self_improve/contracts/r6_model_chooser_prereg.yaml")
            for sample_index in range(8):
                specs.append({**window, "window_index": window_index, "chooser_id": "base",
                              "chooser_kind": "base", "adapter_path": None, "adapter_sha256": None,
                              "query_kind": query_kind, "relabel_index": relabel_index,
                              "sample_index": sample_index,
                              "query_seed": query_seed(window["seed"], window["stage"], sample_index,
                                                       noise=query_kind == "noise"),
                              "prompt": rendered["prompt"],
                              "template_sha256": rendered["template_sha256"],
                              "prompt_sha256": rendered["prompt_sha256"]})
        rendered = render_prompt(
            window["stats"], prereg_path=workspace / "artifacts/self_improve/contracts/r6_model_chooser_prereg.yaml")
        selected = [chooser for chooser in choosers[window["seed"]]
                    if not smoke or chooser["kind"] == "u20"]
        for chooser in selected:
            query_kinds = ("sample",) if smoke else ("sample", "noise")
            for query_kind in query_kinds:
                for sample_index in range(8):
                    specs.append({**window, "window_index": window_index, "chooser_id": chooser["id"],
                                  "chooser_kind": chooser["kind"], "adapter_path": chooser["adapter_path"],
                                  "adapter_sha256": chooser["adapter_sha256"],
                                  "query_kind": query_kind, "relabel_index": 0,
                                  "sample_index": sample_index,
                                  "query_seed": query_seed(window["seed"], window["stage"], sample_index,
                                                           noise=query_kind == "noise"),
                                  "prompt": rendered["prompt"],
                                  "template_sha256": rendered["template_sha256"],
                                  "prompt_sha256": rendered["prompt_sha256"]})
    expected = 16 if smoke else 2304
    if len(specs) != expected:
        raise AssertionError(f"expected {expected} queries, got {len(specs)}")
    return specs


def _generate_records(specs: Sequence[dict[str, Any]], backend: Backend) -> list[dict[str, Any]]:
    by_adapter: dict[str | None, list[int]] = defaultdict(list)
    for index, spec in enumerate(specs):
        by_adapter[spec["adapter_path"]].append(index)
    completions = [""] * len(specs)
    for adapter_path, indices in by_adapter.items():
        generated = backend.generate([specs[index]["prompt"] for index in indices],
                                     [specs[index]["query_seed"] for index in indices], adapter_path)
        if len(generated) != len(indices):
            raise RuntimeError("backend returned the wrong completion count")
        for index, completion in zip(indices, generated, strict=True):
            completions[index] = completion
    records = []
    for spec, completion in zip(specs, completions, strict=True):
        parsed = parse_completion(completion)
        record = {key: value for key, value in spec.items() if key != "prompt"}
        record.update(raw_completion=completion, parsed_values=parsed["values"], q_j=parsed["q"],
                      valid=parsed["valid"], unnormalised=parsed["unnormalised"])
        records.append(record)
    return records


def _groups(records: Sequence[dict[str, Any]]) -> dict[tuple[int, str, int, str, str], dict[str, Any]]:
    completions: dict[tuple[int, str, int, str, str], list[str]] = defaultdict(list)
    for record in records:
        key = (record["window_index"], record["arm"], record["stage"],
               record["chooser_id"], record["query_kind"])
        completions[key].append(record["raw_completion"])
    if any(len(values) != 8 for values in completions.values()):
        raise ValueError("every saved query group must contain exactly eight completions")
    return {key: aggregate_samples(values) for key, values in completions.items()}


def _chooser_rates(records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    validity, unnormalised = {}, {}
    for chooser_id in sorted({record["chooser_id"] for record in records}):
        subset = [record for record in records if record["chooser_id"] == chooser_id]
        valid_count = sum(record["valid"] for record in subset)
        validity[chooser_id] = {"valid": valid_count, "total": len(subset),
                                "rate": valid_count / len(subset)}
        unnormalised[chooser_id] = sum(record["unnormalised"] for record in subset) / len(subset)
    return validity, unnormalised


def run_g1(workspace: Path, backend: Backend, output_dir: Path, *,
           choosers: Mapping[int, Sequence[Mapping[str, str]]],
           deterministic_wall_time: bool = False, began: float | None = None) -> dict[str, Any]:
    began = time.monotonic() if began is None else began
    records = _generate_records(query_specs(workspace, choosers), backend)
    aggregates = _groups(records)
    validity, unnormalised = _chooser_rates(records)
    sensitivity = []
    windows = sorted({(record["window_index"], record["arm"], record["stage"]) for record in records})
    for window_index, arm, stage in windows:
        base = aggregates[(window_index, arm, stage, "base", "sample")]["q"]
        for relabel_index in (1, 2):
            relabelled = aggregates[(window_index, arm, stage, "base", f"relabel_{relabel_index}")]["q"]
            sensitivity.append(tv_distance(base, relabelled))
    input_sensitivity = sum(sensitivity) / len(sensitivity)
    sample_groups = {key: value for key, value in aggregates.items() if key[-1] == "sample"}
    chooser_ids = sorted(validity)
    fallback_counts = {chooser_id: sum(value["fallback"] for key, value in sample_groups.items()
                                               if key[3] == chooser_id)
                       for chooser_id in chooser_ids}
    adapter_hashes = {chooser_id: next(record["adapter_sha256"] for record in records
                                       if record["chooser_id"] == chooser_id)
                      for chooser_id in chooser_ids}
    all_valid = all(value["rate"] >= 0.95 for value in validity.values())
    metrics = {
        "schema": "r6-g1-offline-replay-v1", "backend": type(backend).__name__, "windows": 36,
        "sample_level_validity": validity, "unnormalised_rate": unnormalised,
        "input_sensitivity": input_sensitivity, "fallback_stage_counts": fallback_counts,
        "sample_stage_count": len(sample_groups), "adapter_sha256": adapter_hashes,
        "query_count": {"total": len(records), "formula": QUERY_FORMULA},
        "G1": {"validity_pass": all_valid, "input_sensitivity_pass": input_sensitivity >= 0.10,
               "pass": all_valid and input_sensitivity >= 0.10},
        "wall_time_seconds": 0.0 if deterministic_wall_time else time.monotonic() - began,
        "wall_time_measurement": ("deterministic zero for mock dry-run"
                                  if deterministic_wall_time else "monotonic elapsed time including model boot"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "completions.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                for record in records), encoding="utf-8")
    (output_dir / "g1_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return metrics


def run_smoke(workspace: Path, backend: Backend, output_dir: Path, *,
              choosers: Mapping[int, Sequence[Mapping[str, str]]], began: float | None = None,
              deterministic_wall_time: bool = False) -> dict[str, Any]:
    began = time.monotonic() if began is None else began
    records = _generate_records(query_specs(workspace, choosers, smoke=True), backend)
    elapsed = 0.0 if deterministic_wall_time else time.monotonic() - began
    metrics = {"schema": "r6-smoke-v1", "query_count": len(records),
               "window": {"seed": 137, "arm": "failure_driven", "stage": 1},
               "choosers": ["base", "u20_137"], "elapsed_seconds": elapsed,
               "adapter_sha256": {record["chooser_id"]: record["adapter_sha256"] for record in records}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "completions.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                for record in records), encoding="utf-8")
    (output_dir / "smoke_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for index, record in enumerate(records, 1):
        print(f"--- completion {index}/16 ---")
        print(record["raw_completion"])
    print(json.dumps({"elapsed_seconds": elapsed, "query_count": len(records)}, sort_keys=True))
    return metrics


def run_g2(completions_path: Path, output_dir: Path) -> dict[str, Any]:
    gate_path = completions_path.parent / "g1_metrics.json"
    if not gate_path.is_file():
        raise RuntimeError(f"G2 refused: missing sibling G1 metrics: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("G1", {}).get("pass") is not True:
        raise RuntimeError("G2 refused: G1.pass is not true")
    records = [json.loads(line) for line in completions_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 2304:
        raise ValueError(f"G2 requires exactly 2304 saved completion records, got {len(records)}")
    aggregates = _groups(records)
    windows = sorted({(record["window_index"], record["seed"], record["arm"], record["stage"])
                      for record in records})
    if len(windows) != 36:
        raise ValueError(f"G2 requires 36 saved windows, got {len(windows)}")
    noise_values = []
    identity: dict[str, dict[int, list[float]]] = {"u20": defaultdict(list), "u80": defaultdict(list)}
    for window_index, seed, arm, stage in windows:
        base = aggregates[(window_index, arm, stage, "base", "sample")]["q"]
        noise = aggregates[(window_index, arm, stage, "base", "noise")]["q"]
        noise_values.append(tv_distance(base, noise))
        for kind in ("u20", "u80"):
            adapted = aggregates[(window_index, arm, stage, f"{kind}_{seed}", "sample")]["q"]
            identity[kind][seed].append(tv_distance(adapted, base))
    identity_metrics = {}
    for kind, per_seed in identity.items():
        seed_means = {str(seed): canonical_metric(sum(values) / len(values))
                      for seed, values in sorted(per_seed.items())}
        all_values = [value for values in per_seed.values() for value in values]
        identity_metrics[kind] = {
            "per_seed": seed_means,
            "pooled": canonical_metric(sum(all_values) / len(all_values)),
        }
    tv_noise = canonical_metric(sum(noise_values) / len(noise_values))
    pooled = identity_metrics["u80"]["pooled"]
    source_sha256 = hashlib.sha256(completions_path.read_bytes()).hexdigest()
    metrics = {"schema": "r6-g2-offline-replay-v1", "source_completions_sha256": source_sha256,
               "TV_identity": identity_metrics, "TV_noise": tv_noise,
               "G2": {"identity_threshold_pass": pooled >= 0.10,
                      "noise_ratio_pass": pooled >= 2 * tv_noise,
                      "pass": pooled >= 0.10 and pooled >= 2 * tv_noise}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "g2_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("g1", "g2"), default="g1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backend", choices=("mock", "vllm"), default="mock")
    parser.add_argument("--mode", choices=("constant", "proportional", "garbage"), default="proportional")
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-map", type=Path)
    parser.add_argument("--completions", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/self_improve/r6/replay_dryrun"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[2]
    if args.phase == "g2":
        if args.smoke or args.completions is None:
            parser.error("--phase g2 requires --completions and cannot use --smoke")
        print(json.dumps(run_g2(args.completions, args.output_dir), indent=2, sort_keys=True))
        return
    if args.completions is not None:
        parser.error("--completions is only valid with --phase g2")
    choosers = load_adapter_map(args.adapter_map) if args.adapter_map else default_choosers(workspace)
    began = time.monotonic()
    if args.backend == "mock":
        backend: Backend = MockBackend(args.mode)
    else:
        if not args.model_path:
            parser.error("--backend vllm requires --model-path")
        backend = VllmBackend(args.model_path)
    if args.smoke:
        run_smoke(workspace, backend, args.output_dir, choosers=choosers, began=began,
                  deterministic_wall_time=args.backend == "mock")
    else:
        metrics = run_g1(workspace, backend, args.output_dir, choosers=choosers, began=began,
                         deterministic_wall_time=args.backend == "mock")
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
