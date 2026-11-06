import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from praxis.io import read_json, write_json
from praxis.rules import load_rule, apply_rule
from praxis.runner import run


def fixture(tmp_path, source):
    data = tmp_path/'data'
    dataset = data/'sample'
    write_json(dataset/'meta/info.json', {'fps': 10, 'chunks_size': 1000,
        'data_path': 'data/episode_{episode_index:06d}.parquet',
        'features': {'observation.state': {'shape': [1], 'names': ['gripper']}}})
    (dataset/'data').mkdir()
    for eid in [2, 3]:
        pq.write_table(pa.table({'observation.state': [[0.], [0.], [1.], [1.]],
            'frame_index': [0, 1, 2, 3], 'timestamp': [0., .1, .2, .3],
            'episode_index': [eid]*4}), dataset/f'data/episode_{eid:06d}.parquet')
    rules = tmp_path/'rules'
    (rules/'task').mkdir(parents=True)
    (rules/'task/rule.py').write_text(source)
    split = tmp_path/'tasks.json'
    write_json(split, {'tasks': [{'task_id': 'task', 'dataset': 'sample', 'target_ids': [2, 3]}]})
    return data, split, rules, tmp_path/'output'


def test_new_task_runs_without_gt_or_exemplar_count(tmp_path):
    args = fixture(tmp_path, 'def segment(ep):\n    return [2]\n')
    manifest = run(*args)
    assert manifest['prediction_count'] == 2
    assert read_json(args[-1]/'task/episode_000002.json')['cuts'] == [2]
    assert any(p.endswith('rule.py') for p in read_json(args[-1]/'freeze.json')['files'])
    with pytest.raises(FileExistsError):
        run(*args)


def test_initialization_once_and_template_relative_to_rule(tmp_path):
    args = fixture(tmp_path, '''import json
def load(directory):
    class Rule:
        def __init__(self):
            self.cuts = json.loads((directory/'template.json').read_text())
            self.calls = 0
        def segment(self, ep):
            self.calls += 1
            return self.cuts if self.calls == 1 else None
    return Rule()
''')
    write_json(args[2]/'task/template.json', [2])
    assert run(*args)['status_counts'] == {'review': 1, 'abstain': 1}


@pytest.mark.parametrize('value', ['[2, 2]', '[4]', '[1.5]', '(2,)', 'True'])
def test_invalid_rule_output_kept_as_error(tmp_path, value):
    args = fixture(tmp_path, f'def segment(ep):\n    return {value}\n')
    assert run(*args)['status_counts'] == {'error': 2}


def test_empty_cut_list_means_one_segment(tmp_path):
    args = fixture(tmp_path, 'def segment(ep):\n    return []\n')
    run(*args)
    result = read_json(args[-1]/'task/episode_000002.json')
    assert result['has_prediction'] is True and result['cuts'] == []


def test_rule_resource_mutation_does_not_publish_manifest(tmp_path):
    args = fixture(tmp_path, '''from pathlib import Path
def segment(ep):
    Path(__file__).with_name('new_resource.txt').write_text('changed')
    return []
''')
    with pytest.raises(RuntimeError, match='changed during run'):
        run(*args)
    assert not (args[-1]/'manifest.json').exists()


def test_reload_reads_current_source_not_cached_bytecode(tmp_path):
    p=tmp_path/'rule.py'
    p.write_text('def segment(ep): return [1]\n')
    assert load_rule(tmp_path).segment({}) == [1]
    p.write_text('def segment(ep): return [2]\n')
    assert load_rule(tmp_path).segment({}) == [2]
