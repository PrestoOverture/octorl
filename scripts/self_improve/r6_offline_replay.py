#!/usr/bin/env python3
"""Offline G1/G2 replay for the R6 chooser, with mock and vLLM backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from src.curriculum.failure_driven import CELLS, PI0
from src.curriculum.model_chooser import (
    aggregate_samples, cyclic_relabellings, query_seed, render_prompt, tv_distance,
)


SEEDS = (42, 137, 2718)
ARMS = ("fixed", "failure_driven")
STAGES = tuple(range(1, 7))


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
    """Lazy vLLM backend; constructing it is the first heavyweight operation."""

    def __init__(self, model_path: str, **llm_kwargs: Any) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path, trust_remote_code=True, enable_lora=True, **llm_kwargs)

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
            adapter_id = int(hashlib.sha256(adapter_path.encode()).hexdigest()[:8], 16) or 1
            request = LoRARequest(Path(adapter_path).name, adapter_id, adapter_path)
        outputs = self.llm.generate(rendered, sampling_params=params, lora_request=request, use_tqdm=False)
        return [item.outputs[0].text for item in outputs]


def default_choosers(workspace: Path) -> dict[int, list[dict[str, str | None]]]:
    checkpoints = workspace / "artifacts/self_improve/r5/checkpoints"
    result = {}
    for seed in SEEDS:
        u20 = checkpoints / (f"r3c_{seed}_u20" if seed in (42, 137) else "r4r_warmup_2718_u20")
        u80 = checkpoints / f"r4r_failure_driven_{seed}_u80"
        for path in (u20, u80):
            if not path.is_dir():
                raise FileNotFoundError(f"adapter path is unresolved: {path}")
        def adapter(kind: str, path: Path) -> dict[str, str]:
            weights = path / "adapter_model.safetensors"
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            return {"id": f"{kind}_{seed}", "kind": kind,
                    "adapter_path": str(path.resolve()), "adapter_sha256": digest}
        result[seed] = [
            adapter("u20", u20), adapter("u80", u80),
        ]
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


def _query_specs(workspace: Path, choosers: dict[int, list[dict[str, str | None]]] | None = None) -> list[dict[str, Any]]:
    specs = []
    choosers = choosers or default_choosers(workspace)
    for window_index, window in enumerate(load_windows(workspace)):
        variants = [("sample", 0, window["stats"]), ("noise", 0, window["stats"])]
        variants.extend((f"relabel_{index}", index, stats)
                        for index, stats in enumerate(cyclic_relabellings(window["stats"]), 1))
        for query_kind, relabel_index, stats in variants:
            noise = query_kind == "noise"
            rendered = render_prompt(stats, prereg_path=workspace / "artifacts/self_improve/contracts/r6_model_chooser_prereg.yaml")
            for sample_index in range(8):
                specs.append({**window, "window_index": window_index, "chooser_id": "base",
                              "chooser_kind": "base", "adapter_path": None,
                              "query_kind": query_kind, "relabel_index": relabel_index,
                              "sample_index": sample_index,
                              "query_seed": query_seed(window["seed"], window["stage"], sample_index, noise=noise),
                              "prompt": rendered["prompt"],
                              "template_sha256": rendered["template_sha256"],
                              "prompt_sha256": rendered["prompt_sha256"]})
        rendered = render_prompt(window["stats"], prereg_path=workspace / "artifacts/self_improve/contracts/r6_model_chooser_prereg.yaml")
        for chooser in choosers[window["seed"]]:
            for query_kind in ("sample", "noise"):
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
    return specs


def run_replay(workspace: Path, backend: Backend, output_dir: Path, *, deterministic_wall_time: bool = False,
               choosers: dict[int, list[dict[str, str | None]]] | None = None) -> dict[str, Any]:
    began = time.monotonic()
    specs = _query_specs(workspace, choosers)
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
    grouped: dict[tuple[int, str, int, str, str], list[str]] = defaultdict(list)
    for spec, completion in zip(specs, completions, strict=True):
        parsed = __import__("src.curriculum.model_chooser", fromlist=["parse_completion"]).parse_completion(completion)
        record = {key: value for key, value in spec.items() if key != "prompt"}
        record.update(raw_completion=completion, parsed_values=parsed["values"], q_j=parsed["q"],
                      valid=parsed["valid"], unnormalised=parsed["unnormalised"])
        records.append(record)
        grouped[(spec["window_index"], spec["arm"], spec["stage"], spec["chooser_id"],
                 spec["query_kind"])].append(completion)

    aggregates = {key: aggregate_samples(values) for key, values in grouped.items()}
    validity: dict[str, dict[str, int | float]] = {}
    unnormalised: dict[str, float] = {}
    for chooser_id in sorted({record["chooser_id"] for record in records}):
        subset = [record for record in records if record["chooser_id"] == chooser_id]
        valid_count = sum(record["valid"] for record in subset)
        validity[chooser_id] = {"valid": valid_count, "total": len(subset), "rate": valid_count / len(subset)}
        unnormalised[chooser_id] = sum(record["unnormalised"] for record in subset) / len(subset)

    sensitivity_values = []
    noise_values = []
    identity: dict[str, dict[int, list[float]]] = {"u20": defaultdict(list), "u80": defaultdict(list)}
    for window_index, window in enumerate(load_windows(workspace)):
        base = aggregates[(window_index, window["arm"], window["stage"], "base", "sample")]["q"]
        noise = aggregates[(window_index, window["arm"], window["stage"], "base", "noise")]["q"]
        noise_values.append(tv_distance(base, noise))
        for relabel_index in (1, 2):
            relabelled = aggregates[(window_index, window["arm"], window["stage"], "base",
                                     f"relabel_{relabel_index}")]["q"]
            sensitivity_values.append(tv_distance(base, relabelled))
        for kind in ("u20", "u80"):
            adapted = aggregates[(window_index, window["arm"], window["stage"],
                                  f"{kind}_{window['seed']}", "sample")]["q"]
            identity[kind][window["seed"]].append(tv_distance(adapted, base))

    identity_metrics = {}
    for kind, per_seed in identity.items():
        seed_means = {str(seed): sum(values) / len(values) for seed, values in sorted(per_seed.items())}
        all_values = [value for values in per_seed.values() for value in values]
        identity_metrics[kind] = {"per_seed": seed_means, "pooled": sum(all_values) / len(all_values)}
    input_sensitivity = sum(sensitivity_values) / len(sensitivity_values)
    tv_noise = sum(noise_values) / len(noise_values)
    all_valid = all(value["rate"] >= 0.95 for value in validity.values())
    sample_aggregates = [value for key, value in aggregates.items() if key[-1] == "sample"]
    metrics = {
        "schema": "r6-offline-replay-v1", "backend": type(backend).__name__,
        "windows": 36, "sample_level_validity": validity, "unnormalised_rate": unnormalised,
        "input_sensitivity": input_sensitivity, "TV_identity": identity_metrics,
        "TV_noise": tv_noise,
        "G1": {"validity_pass": all_valid, "input_sensitivity_pass": input_sensitivity >= 0.10,
               "pass": all_valid and input_sensitivity >= 0.10},
        "G2": {"pooled_TV_identity": identity_metrics["u80"]["pooled"],
               "TV_noise": tv_noise,
               "pass": identity_metrics["u80"]["pooled"] >= 0.10
                       and identity_metrics["u80"]["pooled"] >= 2 * tv_noise},
        "query_count": {"total": len(records), "formula": "36 * (base: 8+8+2*8; paired U20: 8+8; paired U80: 8+8) = 2304"},
        "fallback_stages": sum(value["fallback"] for value in sample_aggregates),
        "sample_stage_count": len(sample_aggregates),
        "fallback_q_all_pi0": all(value["q"] == PI0 for value in sample_aggregates if value["fallback"]),
        "wall_time_seconds": 0.0 if deterministic_wall_time else time.monotonic() - began,
        "wall_time_measurement": ("deterministic zero for mock dry-run"
                                  if deterministic_wall_time else "monotonic elapsed time"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "completions.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                for record in records), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "vllm"), default="mock")
    parser.add_argument("--mode", choices=("constant", "proportional", "garbage"), default="proportional")
    parser.add_argument("--model-path")
    parser.add_argument("--u20-adapter", action="append", default=[], metavar="SEED=PATH",
                        help="override a default U20 adapter path (repeatable)")
    parser.add_argument("--u80-adapter", action="append", default=[], metavar="SEED=PATH",
                        help="override a default U80 adapter path (repeatable)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/self_improve/r6/replay_dryrun"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[2]
    choosers = default_choosers(workspace)
    for kind, overrides in (("u20", args.u20_adapter), ("u80", args.u80_adapter)):
        for item in overrides:
            seed_text, separator, path_text = item.partition("=")
            if not separator or int(seed_text) not in SEEDS:
                parser.error(f"--{kind}-adapter requires SEED=PATH for one of {SEEDS}")
            seed = int(seed_text)
            path = Path(path_text).expanduser().resolve()
            weights = path / "adapter_model.safetensors"
            if not weights.is_file():
                parser.error(f"adapter weights not found: {weights}")
            replacement = {"id": f"{kind}_{seed}", "kind": kind, "adapter_path": str(path),
                           "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
            choosers[seed] = [replacement if chooser["kind"] == kind else chooser
                              for chooser in choosers[seed]]
    if args.backend == "mock":
        backend: Backend = MockBackend(args.mode)
    else:
        if not args.model_path:
            parser.error("--backend vllm requires --model-path")
        backend = VllmBackend(args.model_path)
    metrics = run_replay(workspace, backend, args.output_dir,
                         deterministic_wall_time=args.backend == "mock", choosers=choosers)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
