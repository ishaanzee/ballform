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
    track_id: int | None = None
    appearance: tuple[float, float, float] | None = None


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
    # Source-video frame where the outcome was observed.  It is deliberately
    # separate from end_s: the visible flight may continue after a rim event.
    outcome_frame: int | None = None
    # Game mode: "jump shot", "floater", "layup", "dunk", "layup or dunk" or "tip".
    shot_type: str | None = None
    # Game mode: how the attempt was found (path, contact frame, last-handler
    # shooter track) for shots not established by raised-hand release contact.
    attempt: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)
