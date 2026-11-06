"""Execute fixed task rules, preserving every target and immutable run outputs."""
from collections import Counter
from pathlib import Path
import platform
import importlib.metadata
import pyarrow.parquet as pq
from .data import load_episode, resolve_source
from .io import inside, read_json, write_json, sha256
from .rules import R2Rule


def run(data_root, split_path, rules_root, output):
    data_root, split_path, rules_root, output = map(Path, (data_root, split_path, rules_root, output))
    split = read_json(split_path)
    ids = set()
    for task in split["tasks"]:
        if task["task_id"] in ids:
            raise ValueError("duplicate task")
        ids.add(task["task_id"])
        targets = task["target_ids"]
        if len(set(targets)) != len(targets) or set(targets) & set(task["exemplar_ids"]):
            raise ValueError("duplicate targets or exemplar leakage")
    # Preflight task resources before creating a run. No test signals are read.
    rules = {t["task_id"]: R2Rule(inside(rules_root, t["task_id"])) for t in split["tasks"]}
    for t in split["tasks"]:
        if rules[t["task_id"]].parameters["dataset"] != t["dataset"]:
            raise ValueError("rule dataset mismatch")
    files = [p for p in rules_root.rglob("*") if p.is_file() and p.suffix in (".json", ".npz")]
    files += list(Path(__file__).parent.rglob("*.py"))
    hashes = {str(p.resolve()): sha256(p) for p in files}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "freeze.json", {"files": hashes, "split_sha256": sha256(split_path),
        "python": platform.python_version(), "dependencies": {k: importlib.metadata.version(k)
        for k in ("numpy", "scipy", "numba", "pyarrow")}})
    records, counts = [], Counter()
    for task in split["tasks"]:
        rule = rules[task["task_id"]]
        for eid in task["target_ids"]:
            n, source_hash = 0, None
            try:
                source = resolve_source(inside(data_root, task["dataset"]), eid)
                source_hash = sha256(source)
                n = pq.read_metadata(source).num_rows
                target, _ = load_episode(source, rule.parameters["channels"], eid, rule.parameters["required_columns"])
                n = target["n"]
                result = rule.segment(target, dataset=task["dataset"], episode_id=eid)
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
                "prediction_path": rel, "prediction_sha256": sha256(output / rel), "physical_source_sha256": source_hash})
            counts[result["status"]] += 1
        print(f"{task['task_id']}: {len(task['target_ids'])} episodes", flush=True)
    if any(sha256(path) != h for path, h in hashes.items()):
        raise RuntimeError("rule/code changed during run")
    if sha256(split_path) != read_json(output / "freeze.json")["split_sha256"]:
        raise RuntimeError("split changed during run")
    manifest = {"records": records, "prediction_count": len(records), "status_counts": dict(counts),
                "split_sha256": sha256(split_path), "freeze_sha256": sha256(output / "freeze.json")}
    write_json(output / "manifest.json", manifest)
    return manifest
