import pytest
from praxis.boundaries import from_intervals, to_intervals
from praxis.evaluation import match_segments, aggregate
from praxis.io import inside


def test_boundary_roundtrip_and_one_segment():
    assert from_intervals(to_intervals([30, 70], 100), 100) == [30, 70]
    assert to_intervals([], 1) == [{'start_frame': 0, 'end_frame': 1}]
    with pytest.raises(ValueError):
        from_intervals([], 100)


@pytest.mark.parametrize('cuts', [[0], [100], [30, 30], [70, 30], [True], [2.5]])
def test_invalid_boundaries(cuts):
    with pytest.raises(ValueError):
        to_intervals(cuts, 100)


def test_no_gap_or_overlap():
    with pytest.raises(ValueError):
        from_intervals([{'start_frame': 0, 'end_frame': 3}, {'start_frame': 4, 'end_frame': 9}], 9)


def test_matching_and_failure_denominator():
    gt = to_intervals([50], 100)
    assert len(match_segments(to_intervals([55], 100), gt, .75)) == 2
    assert len(match_segments([gt[0], gt[0]], gt, .75)) == 1
    result = aggregate([{'predicted_segments': 0, 'truth_segments': 2, 'tp50': 0, 'tp75': 0}])
    assert result['f1_75'] == 0


def test_path_escape(tmp_path):
    with pytest.raises(ValueError):
        inside(tmp_path, '../outside')
