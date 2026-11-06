"""Execute fixed task rules, preserving every target and immutable run outputs."""
from collections import Counter
from pathlib import Path
import platform
import importlib.metadata
import pyarrow.parquet as pq
from .data import load_episode, resolve_source
from .io import inside, read_json, write_json, sha256
from .rules import load_rule, apply_rule


def run(data_root, split_path, rules_root, output):
    data_root, split_path, rules_root, output = map(Path, (data_root, split_path, rules_root, output))
    split = read_json(split_path)
    if output.exists():
        raise FileExistsError("refusing run overwrite")
    ids = set()
    episode_keys = set()
    for task in split["tasks"]:
        if task["task_id"] in ids:
            raise ValueError("duplicate task")
        ids.add(task["task_id"])
        targets = task["target_ids"]
        if any(type(e) is not int or e < 0 for e in targets):
            raise ValueError("invalid episode ID")
        for eid in targets:
            key = (task["dataset"], eid)
            if key in episode_keys:
                raise ValueError("duplicate dataset episode")
            episode_keys.add(key)
        if len(set(targets)) != len(targets) or set(targets) & set(task.get("exemplar_ids", [])):
            raise ValueError("duplicate targets or exemplar leakage")
    # Freeze all task assets before loading trusted rule code; ignore only caches.
    def snapshot():
        files = [p for p in rules_root.rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")]
        files += list(Path(__file__).parent.rglob("*.py"))
        return {str(p.resolve()): sha256(p) for p in files}
    hashes = snapshot()
    split_hash = sha256(split_path)
    rules = {t["task_id"]: load_rule(inside(rules_root, t["task_id"])) for t in split["tasks"]}
    for t in split["tasks"]:
        parameters = getattr(rules[t["task_id"]], "parameters", {})
        if "dataset" in parameters and parameters["dataset"] != t["dataset"]:
            raise ValueError("rule dataset mismatch")
    if snapshot() != hashes:
        raise RuntimeError("rule files changed during loading")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "freeze.json", {"files": hashes, "split_sha256": split_hash,
        "task_list": split, "data_root": str(data_root.resolve()),
        "rules_root": str(rules_root.resolve()),
        "python": platform.python_version(), "dependencies": {k: importlib.metadata.version(k)
        for k in ("numpy", "scipy", "numba", "pyarrow")},
        "installed_distributions": sorted({(d.metadata['Name'], d.version)
            for d in importlib.metadata.distributions() if d.metadata['Name']})})
    records, counts = [], Counter()
    for task in split["tasks"]:
        rule = rules[task["task_id"]]
        for eid in task["target_ids"]:
            n, source_hash, metadata_hash = 0, None, None
            try:
                dataset_path = inside(data_root, task["dataset"])
                metadata_hash = sha256(dataset_path / "meta/info.json")
                source = resolve_source(dataset_path, eid)
                source_hash = sha256(source)
                n = pq.read_metadata(source).num_rows
                channels = getattr(rule, "channels", None)
                if channels is None:
                    info = read_json(inside(data_root, task["dataset"]) / "meta/info.json")
                    channels = {"fps": info["fps"], "features": info["features"]}
                columns = getattr(rule, "required_columns", ["observation.state"])
                if any(not (c == "action" or c == "observation.state" or c.startswith("observation.state.")) for c in columns):
                    raise ValueError("only physical columns may be requested")
                target, _ = load_episode(source, channels, eid, columns)
                n = target["n"]
                result = apply_rule(rule, target, task["dataset"], eid)
            except Exception as exc:
                category = type(exc).__name__
                result = {"dataset": task["dataset"], "episode_id": eid, "frame_count": n,
                    "status": "error", "cuts": [], "has_prediction": False,
                    "review_reasons": ["execution_failed"],
                    "error": {"category": category, "code": str(exc) if category == "InputContractError" else category,
                              "frame_count_known": n > 0}}
            rel = f"{task['task_id']}/episode_{eid:06d}.json"
            write_json(output / rel, result)
            records.append({"task_id": task["task_id"], "dataset": task["dataset"], "episode_id": eid,
                "prediction_path": rel, "prediction_sha256": sha256(output / rel),
                "physical_source_sha256": source_hash, "input_metadata_sha256": metadata_hash})
            counts[result["status"]] += 1
        print(f"{task['task_id']}: {len(task['target_ids'])} episodes", flush=True)
    if snapshot() != hashes:
        raise RuntimeError("rule/code changed during run")
    if sha256(split_path) != read_json(output / "freeze.json")["split_sha256"]:
        raise RuntimeError("split changed during run")
    manifest = {"records": records, "prediction_count": len(records), "status_counts": dict(counts),
                "split_sha256": sha256(split_path), "freeze_sha256": sha256(output / "freeze.json")}
    write_json(output / "manifest.json", manifest)
    return manifest
