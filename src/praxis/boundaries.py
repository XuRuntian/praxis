"""Only this module defines conversion between cuts and half-open intervals.

A cut k means the next segment starts at original zero-based frame k.
Cuts exclude the implicit episode endpoints 0 and frame_count.
"""


def validate_cuts(cuts, frame_count):
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if any(type(k) is not int or not 0 < k < frame_count for k in cuts):
        raise ValueError("cuts must be integer internal frame boundaries")
    if any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValueError("cuts must be strictly increasing")


def to_intervals(cuts, frame_count):
    validate_cuts(cuts, frame_count)
    edges = [0, *cuts, frame_count]
    return [{"start_frame": a, "end_frame": b} for a, b in zip(edges, edges[1:])]


def from_intervals(segments, frame_count):
    if not segments:
        raise ValueError("empty prediction is not a one-segment prediction")
    cursor = 0
    for segment in segments:
        a, b = segment["start_frame"], segment["end_frame"]
        if type(a) is not int or type(b) is not int or a != cursor or not a < b <= frame_count:
            raise ValueError("intervals must cover the episode without gaps or overlaps")
        cursor = b
    if cursor != frame_count:
        raise ValueError("incomplete coverage")
    cuts = [s["start_frame"] for s in segments[1:]]
    validate_cuts(cuts, frame_count)
    return cuts
