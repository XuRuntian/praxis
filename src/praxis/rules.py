"""Minimal task-rule interface; R2 is an adapter, not a mandatory algorithm."""
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from .boundaries import from_intervals, validate_cuts
from .io import read_json
from .methods.r2.io_contract import load_sources
from .methods.r2.method_runtime import predict


class SegmentationRule(Protocol):
    def segment(self, episode: dict) -> list[int] | None | dict: ...


class R2Rule:
    def __init__(self, directory):
        self.parameters = read_json(directory / "parameters.json")
        self.sources = load_sources(directory, self.parameters)
        self.channels = self.parameters["channels"]
        self.required_columns = self.parameters["required_columns"]

    def segment(self, episode):
        dataset, episode_id = self.parameters["dataset"], episode["eid"]
        result = predict(self.sources, episode, self.parameters, dataset, episode_id, "r2")
        valid = bool(result["segments"])
        return {
            "dataset": dataset, "episode_id": episode_id,
            "frame_count": result["frame_count"], "status": result["status"],
            "cuts": from_intervals(result["segments"], result["frame_count"]) if valid else [],
            "has_prediction": valid,
            "review_reasons": result["review_reasons"],
            "error": result["error"],
        }


def load_rule(directory):
    """Load trusted local rule.py; no registry or dependency on task names."""
    directory = Path(directory).resolve()
    path = directory / "rule.py"
    # Compile directly: avoid stale bytecode when an agent edits a rule rapidly.
    namespace = {"__file__": str(path), "__name__": "praxis_task_rule"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    rule = namespace["load"](directory) if "load" in namespace else SimpleNamespace(**namespace)
    if not callable(getattr(rule, "segment", None)):
        raise ValueError("rule.py must provide segment(episode) or load(directory)")
    return rule


def apply_rule(rule, episode, dataset, episode_id):
    """Normalize a minimal cut list, or validate a richer compatibility result."""
    value = rule.segment(episode)
    result = dict(value) if isinstance(value, dict) else {
        "dataset": dataset, "episode_id": episode_id, "frame_count": episode["n"],
        "cuts": value if value is not None else [], "has_prediction": value is not None,
        "status": "review" if value is not None else "abstain",
        "review_reasons": ["rule_output_requires_review"] if value is not None else ["rule_abstained"],
        "error": None,
    }
    result["review_reasons"] = list(result["review_reasons"]) + episode.get("input_notes", [])
    if (result["dataset"] != dataset or result["episode_id"] != episode_id or
            result["frame_count"] != episode["n"]):
        raise ValueError("rule changed episode identity or frame count")
    if type(result["has_prediction"]) is not bool or not isinstance(result["cuts"], list):
        raise ValueError("rule must return a list of cuts or None")
    if not isinstance(result["review_reasons"], list) or any(not isinstance(x, str) for x in result["review_reasons"]):
        raise ValueError("invalid review reasons")
    if result["has_prediction"]:
        if result["status"] not in ("ok", "review") or result["error"] is not None:
            raise ValueError("inconsistent prediction status")
        validate_cuts(result["cuts"], episode["n"])
    elif result["status"] != "abstain" or result["cuts"] or result["error"] is not None:
        raise ValueError("invalid abstention")
    return result
