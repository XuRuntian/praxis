"""Independent segmentation scoring. Ground truth is read only here."""
from pathlib import Path
from .boundaries import to_intervals
from .io import inside, read_json, sha256


def iou(a, b):
    intersection = max(0, min(a['end_frame'], b['end_frame']) - max(a['start_frame'], b['start_frame']))
    union = max(a['end_frame'], b['end_frame']) - min(a['start_frame'], b['start_frame'])
    return intersection / union if union > 0 else 0.


def match_segments(predicted, truth, threshold):
    neighbors = [sorted((j for j, g in enumerate(truth) if iou(p, g) >= threshold),
                        key=lambda j: (-iou(p, truth[j]), j)) for p in predicted]
    owners = {}
    def augment(i, seen):
        for j in neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or augment(owners[j], seen):
                owners[j] = i
                return True
        return False
    for i in range(len(predicted)):
        augment(i, set())
    return sorted((i, j) for j, i in owners.items())


def verified_records(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'manifest.json')
    if sha256(run_dir / 'freeze.json') != manifest['freeze_sha256']:
        raise ValueError('freeze changed')
    records = manifest['records']
    keys = [(r['dataset'], r['episode_id']) for r in records]
    if len(set(keys)) != len(keys) or len(keys) != manifest['prediction_count']:
        raise ValueError('incomplete or duplicate predictions')
    for row in records:
        path = inside(run_dir, row['prediction_path'])
        if sha256(path) != row['prediction_sha256']:
            raise ValueError('prediction changed')
        yield row, read_json(path)


def aggregate(rows):
    result = {k: sum(r[k] for r in rows) for k in ('predicted_segments', 'truth_segments', 'tp50', 'tp75')}
    den = result['predicted_segments'] + result['truth_segments']
    result.update(f1_50=2*result['tp50']/den if den else 0., f1_75=2*result['tp75']/den if den else 0.)
    return result


def score(run_dir, data_root):
    records = list(verified_records(run_dir))  # Verify every prediction before reading any truth.
    datasets, tasks = {}, {}
    for row, prediction in records:
        name = row['dataset']
        if name not in datasets:
            payload = read_json(inside(data_root, name) / 'lite_annotations/annotation_schema_v1.json')
            datasets[name] = {e['id']: e for e in payload['episode_annotation']}
        truth = datasets[name][row['episode_id']]
        segments = []
        if prediction['has_prediction']:
            if prediction['frame_count'] != truth['frame_count']:
                raise ValueError('prediction/truth frame mismatch')
            segments = to_intervals(prediction['cuts'], prediction['frame_count'])
        gt = truth['subtasks']
        value = {'predicted_segments': len(segments), 'truth_segments': len(gt),
                 'tp50': len(match_segments(segments, gt, .5)), 'tp75': len(match_segments(segments, gt, .75))}
        tasks.setdefault(name, []).append(value)
    return {'micro': aggregate([r for rows in tasks.values() for r in rows]),
            'tasks': {name: aggregate(rows) for name, rows in tasks.items()}}


def compare(run_dir, reference_root, reference_manifest):
    """Compare frozen predictions, never reading ground truth."""
    from .boundaries import from_intervals
    reference = read_json(reference_manifest)
    index = {(r['task_id'], r['episode_id']): r for r in reference['records']}
    records = list(verified_records(run_dir))
    if {(r['task_id'], r['episode_id']) for r, _ in records} != set(index):
        raise ValueError('episode sets differ')
    differences = []
    for row, result in records:
        old_row = index[(row['task_id'], row['episode_id'])]
        old_path = inside(reference_root, old_row['prediction_path'])
        if sha256(old_path) != old_row['prediction_sha256']:
            raise ValueError('reference prediction hash mismatch')
        old = read_json(old_path)
        expected = {k: old[k] for k in ('dataset', 'episode_id', 'status', 'frame_count', 'error', 'review_reasons')}
        expected['has_prediction'] = bool(old['segments'])
        expected['cuts'] = from_intervals(old['segments'], old['frame_count']) if old['segments'] else []
        if result != expected or row['physical_source_sha256'] != old_row['physical_source_sha256']:
            differences.append({'task_id': row['task_id'], 'episode_id': row['episode_id']})
    return {'compared': len(records), 'identical': len(records)-len(differences), 'differences': differences}
