from pathlib import Path

from slinky_lab.schemas import Frame, RunConfig
from slinky_lab.storage import DataStore, TrajectoryWriter, read_frames


def test_run_index_and_recovery(tmp_path: Path):
    store = DataStore(tmp_path)
    run = store.create_run(RunConfig())
    assert run["status"] == "queued"
    store.update_run(run["run_id"], status="running")
    recovered = DataStore(tmp_path)
    assert recovered.get_run(run["run_id"])["status"] == "interrupted"


def test_hdf5_trajectory_round_trip(tmp_path: Path):
    path = tmp_path / "trajectory.h5"
    frame = Frame(
        time=0.1,
        positions=[[0.0, 0.0, 0.5]],
        quaternions=[[1.0, 0.0, 0.0, 0.0]],
        metrics={"height": 0.5},
    )
    with TrajectoryWriter(path) as writer:
        writer.append(frame)
    frames, total = read_frames(path)
    assert total == 1
    assert frames[0]["time"] == 0.1
    assert frames[0]["metrics"]["height"] == 0.5


def test_hdf5_reader_can_follow_open_writer(tmp_path: Path):
    path = tmp_path / "live.h5"
    frame = Frame(time=0.1, positions=[[0.0, 0.0, 0.5]], quaternions=[[1.0, 0.0, 0.0, 0.0]])
    writer = TrajectoryWriter(path)
    try:
        writer.append(frame)
        frames, total = read_frames(path)
        assert total == 1
        assert frames[0]["time"] == 0.1
    finally:
        writer.close()


def test_read_frames_limit_none_reads_long_trajectory(tmp_path: Path):
    path = tmp_path / "long.h5"
    frame = Frame(time=0.1, positions=[[0.0, 0.0, 0.5]], quaternions=[[1.0, 0.0, 0.0, 0.0]])
    with TrajectoryWriter(path) as writer:
        for index in range(1001):
            frame.time = index * 0.01
            writer.append(frame)
    frames, total = read_frames(path, limit=None)
    assert total == 1001
    assert len(frames) == 1001
