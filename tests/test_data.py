import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from praxis.data import InputContractError, load_episode, resolve_source
from praxis.io import write_json
from praxis.runner import run


def test_reads_only_requested_physical_columns(tmp_path):
    path = tmp_path / 'episode.parquet'
    pq.write_table(pa.table({'observation.state': [[1.], [2.]], 'frame_index': [0, 1],
        'timestamp': [0., .1], 'episode_index': [3, 3],
        'subtask_annotation': ['SECRET', 'SECRET'], 'task_index': [99, 99]}), path)
    channels = {'fps': 10, 'features': {'observation.state': {'shape': [1]}}}
    result, mapping = load_episode(path, channels, 3, ['observation.state'])
    assert list(result['arrays']) == ['observation.state']
    assert len(mapping) == 2
    # R2 filters mismatched episode IDs; preserve its error rather than remap.
    with pytest.raises(InputContractError, match='too_few_frames'):
        load_episode(path, channels, 4, ['observation.state'])


def test_missing_source_does_not_guess_another_id(tmp_path):
    write_json(tmp_path/'meta/info.json', {'data_path': 'data/episode_{episode_index:06d}.parquet', 'chunks_size': 1000})
    with pytest.raises(FileNotFoundError):
        resolve_source(tmp_path, 3)


def test_rejects_exemplar_target_overlap_before_execution(tmp_path):
    split = tmp_path/'split.json'
    write_json(split, {'tasks': [{'task_id': 't', 'dataset': 'sample', 'exemplar_ids': [0, 1], 'target_ids': [1]}]})
    with pytest.raises(ValueError, match='exemplar leakage'):
        run(tmp_path, split, tmp_path/'rules', tmp_path/'output')
    assert not (tmp_path/'output').exists()
