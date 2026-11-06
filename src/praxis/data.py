"""Current LeRobot layout only. No label reads or implicit signal conversion."""
from .io import inside, read_json
from .methods.r2.io_contract import InputContractError, load_target


def resolve_source(dataset, episode_id):
    info = read_json(dataset / "meta/info.json")
    relative = info["data_path"].format(
        episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id)
    path = inside(dataset, relative)
    if path.is_file():
        return path
    matches = list((dataset / "data").glob(f"**/episode_{episode_id:06d}.parquet"))
    if len(matches) == 1:
        return inside(dataset, matches[0].relative_to(dataset))
    if len(matches) > 1:
        raise ValueError("ambiguous_episode_physical_path")
    raise FileNotFoundError("episode_physical_file_missing")


def load_episode(path, channels, episode_id, required_columns):
    # Preserve R2 identity filtering and timestamp handling for exact replay.
    return load_target(path, channels, episode_id, required_columns)
