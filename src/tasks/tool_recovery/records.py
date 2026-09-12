"""Serializable records for the tool-recovery environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Instance:
    task_id: str
    seed: int
    family: str
    fault_type: str
    faulted_field: str | None
    golden_config: dict[str, Any]
    presented_config: dict[str, Any]
    external_state: dict[str, Any]
    fingerprint: str
    generator_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RejectionRecord:
    slot: int
    fault_type: str
    seeds: tuple[int, ...]
    reason: str


@dataclass
class GenerationManifest:
    split: str
    requested_count: int
    start_seed: int
    generator_version: str
    instances: list[Instance] = field(default_factory=list)
    rejections: list[RejectionRecord] = field(default_factory=list)
    duplicate_seeds: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def generated_count(self) -> int:
        return len(self.instances)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryStep:
    index: int
    action_type: str
    tool_name: str | None
    arguments: dict[str, Any] | None
    observation: dict[str, Any] | None


@dataclass
class TrajectoryRecord:
    task_id: str
    instance_seed: int
    model_sampling_seed: int | None
    generator_version: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    reward: int | None = None
    termination_reason: str | None = None
    diagnostic_flags: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["diagnostic_flags"] = sorted(self.diagnostic_flags)
        return data
