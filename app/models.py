from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Detection:
    frame: int
    time_s: float
    x: float
    y: float
    confidence: float
    radius: float = 0.0


@dataclass
class PoseFrame:
    frame: int
    time_s: float
    landmarks: dict[str, tuple[float, float, float]]


@dataclass
class ShotResult:
    number: int
    start_s: float
    release_s: float
    end_s: float
    outcome: str
    outcome_confidence: float
    evidence: list[str]
    metrics: dict[str, float | str | None]
    cues: list[str] = field(default_factory=list)
    game: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)
