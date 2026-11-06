"""Minimal task-rule interface; R2 is an adapter, not a mandatory algorithm."""
from typing import Protocol
from .boundaries import from_intervals
from .io import read_json
from .methods.r2.io_contract import load_sources
from .methods.r2.method_runtime import predict


class SegmentationRule(Protocol):
    def segment(self, episode: dict, *, dataset: str, episode_id: int) -> dict: ...


class R2Rule:
    def __init__(self, directory):
        self.parameters = read_json(directory / "parameters.json")
        self.sources = load_sources(directory, self.parameters)

    def segment(self, episode, *, dataset, episode_id):
        result = predict(self.sources, episode, self.parameters, dataset, episode_id, "r2")
        valid = bool(result["segments"])
        return {
            "dataset": dataset, "episode_id": episode_id,
            "frame_count": result["frame_count"], "status": result["status"],
            "cuts": from_intervals(result["segments"], result["frame_count"]) if valid else [],
            "has_prediction": valid,
            "review_reasons": result["review_reasons"] + episode.get("input_notes", []),
            "error": result["error"],
        }
