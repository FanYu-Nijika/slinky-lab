import json
from pathlib import Path

import h5py

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
        contact_details=[{"position": [0, 0, 0], "geom_a": 1, "geom_b": 2, "normal_force": 0.4,
                          "kind": "stair", "stair_step": 1, "material_index": 0, "surface": "tread"}],
    )
    with TrajectoryWriter(path) as writer:
        writer.append(frame)
    frames, total = read_frames(path)
    assert total == 1
    assert frames[0]["time"] == 0.1
    assert frames[0]["metrics"]["height"] == 0.5
    assert frames[0]["contact_details"][0]["normal_force"] == 0.4
    assert frames[0]["contact_details"][0]["stair_step"] == 1


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


def test_v3_round_trip_grows_variable_contact_arrays_without_truncation(tmp_path: Path):
    path = tmp_path / "large-contacts.h5"
    contact_count = 5000
    contacts = [[float(index), float(index + 1), float(index + 2)] for index in range(contact_count)]
    details = [
        {
            "position": [float(index), float(index + 1), float(index + 2)],
            "geom_a": index,
            "geom_b": index + 1,
            "normal_force": float(index) * 0.25,
            "kind": ("self", "stair", "external")[index % 3],
            "stair_step": index % 6 if index % 3 == 1 else None,
            "material_index": index if index % 2 else None,
            "surface": "tread" if index % 2 else "other",
        }
        for index in range(contact_count)
    ]
    assert len(json.dumps(contacts, separators=(",", ":"))) > 65536
    assert len(json.dumps(details, separators=(",", ":"))) > 524288
    frame = Frame(time=0.1, positions=[[0.0, 0.0, 0.5]], quaternions=[[1.0, 0.0, 0.0, 0.0]],
                  contacts=contacts, contact_details=details, metrics={"height": 0.5})
    with TrajectoryWriter(path) as writer:
        writer.append(frame)
        visible, total = read_frames(path, limit=None)
        assert total == 1
        assert len(visible[0]["contacts"]) == contact_count
        assert len(visible[0]["contact_details"]) == contact_count
        assert visible[0]["contacts"][-1] == contacts[-1]
        assert visible[0]["contact_details"][-1]["geom_b"] == contact_count
        with h5py.File(path, "r", libver="latest", swmr=True) as raw:
            assert raw.attrs["format"] == "slinky-lab-trajectory-v3"
            assert "contacts_json" not in raw
            assert "contact_details_json" not in raw
            assert raw["contacts"].shape == (1, contact_count, 3)
            assert raw["contact_details"].shape == (1, contact_count, 10)
            assert int(raw["contact_counts"][0]) == contact_count
            assert int(raw["contact_detail_counts"][0]) == contact_count
            assert json.loads(raw.attrs["contact_columns"]) == [
                "x", "y", "z", "geom_a", "geom_b", "normal_force", "kind_code",
                "stair_step", "material_index", "surface_code",
            ]


def test_v3_variable_counts_and_committed_frame_barrier(tmp_path: Path):
    path = tmp_path / "variable-contacts.h5"

    def make_frame(time: float, count: int) -> Frame:
        return Frame(time=time, positions=[[0.0, 0.0, 0.5]], quaternions=[[1.0, 0.0, 0.0, 0.0]],
                     contacts=[[float(index), 0.0, 0.0] for index in range(count)])

    writer = TrajectoryWriter(path)
    try:
        writer.append(make_frame(0.1, 0))
        writer.append(make_frame(0.2, 3))
        writer.append(make_frame(0.3, 1))
        frames, total = read_frames(path, limit=None)
        assert total == 3
        assert [len(frame["contacts"]) for frame in frames] == [0, 3, 1]

        # Simulate a writer that has resized and filled a new frame but has
        # not published the committed frame marker yet.
        writer.file["time"].resize((4,))
        writer.file["time"][3] = 0.4
        writer.file["positions"].resize((4, 1, 3))
        writer.file["positions"][3] = [[0.0, 0.0, 0.5]]
        writer.file["quaternions"].resize((4, 1, 4))
        writer.file["quaternions"][3] = [[1.0, 0.0, 0.0, 0.0]]
        writer.file["contacts"].resize((4, writer.file["contacts"].shape[1], 3))
        writer.file["contact_counts"].resize((4,))
        writer.file["contact_counts"][3] = 3
        writer.file["contact_details"].resize((4, writer.file["contact_details"].shape[1], 10))
        writer.file["contact_detail_counts"].resize((4,))
        writer.file["contact_detail_counts"][3] = 0
        writer.file["metrics_json"].resize((4,))
        writer.file["metrics_json"][3] = b"{}"
        writer.file.flush()
        frames, total = read_frames(path, limit=None)
        assert total == 3
        assert len(frames) == 3
    finally:
        writer.close()


def test_read_handwritten_v2_without_contact_details(tmp_path: Path):
    path = tmp_path / "old-v2.h5"
    with h5py.File(path, "w", libver="latest") as raw:
        raw.attrs["format"] = "slinky-lab-trajectory-v2"
        raw.attrs["committed_frames"] = 1
        raw.create_dataset("time", data=[0.25], maxshape=(None,))
        raw.create_dataset("positions", data=[[[0.0, 0.0, 0.5]]], maxshape=(None, 1, 3))
        raw.create_dataset("quaternions", data=[[[1.0, 0.0, 0.0, 0.0]]], maxshape=(None, 1, 4))
        raw.create_dataset("contacts_json", data=[json.dumps([[1.0, 2.0, 3.0]]).encode("utf-8")],
                           dtype="S128", maxshape=(None,))
        raw.create_dataset("metrics_json", data=[json.dumps({"height": 0.5}).encode("utf-8")],
                           dtype="S128", maxshape=(None,))
        raw.flush()
    frames, total = read_frames(path, limit=None)
    assert total == 1
    assert frames[0]["contacts"] == [[1.0, 2.0, 3.0]]
    assert frames[0]["contact_details"] == []
    assert frames[0]["metrics"] == {"height": 0.5}


def test_read_handwritten_v2_with_contact_details(tmp_path: Path):
    path = tmp_path / "old-v2-details.h5"
    detail = {"position": [1.0, 2.0, 3.0], "geom_a": 1, "geom_b": 2, "normal_force": 0.4,
              "kind": "stair", "stair_step": 1, "material_index": 0, "surface": "tread"}
    with h5py.File(path, "w", libver="latest") as raw:
        raw.attrs["format"] = "slinky-lab-trajectory-v2"
        raw.attrs["committed_frames"] = 1
        raw.create_dataset("time", data=[0.25], maxshape=(None,))
        raw.create_dataset("positions", data=[[[0.0, 0.0, 0.5]]], maxshape=(None, 1, 3))
        raw.create_dataset("quaternions", data=[[[1.0, 0.0, 0.0, 0.0]]], maxshape=(None, 1, 4))
        raw.create_dataset("contacts_json", data=[b"[[1.0,2.0,3.0]]"], dtype="S128", maxshape=(None,))
        raw.create_dataset("contact_details_json", data=[json.dumps([detail]).encode("utf-8")],
                           dtype="S512", maxshape=(None,))
        raw.create_dataset("metrics_json", data=[b"{}"], dtype="S128", maxshape=(None,))
        raw.flush()
    frames, total = read_frames(path, limit=None)
    assert total == 1
    assert frames[0]["contact_details"] == [detail]
