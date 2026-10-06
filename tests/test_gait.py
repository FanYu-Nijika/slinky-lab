import pytest

from slinky_lab.gait import GaitRecorder


def feed(recorder, samples):
    for time, forces in samples:
        recorder.update(time, 0.01, forces)
    return recorder.finalize()


def force(step, label, value=2.0):
    return {step: {"first": value if label == "first" else 0.0,
                   "last": value if label == "last" else 0.0,
                   "middle": value if label == "middle" else 0.0}}


def test_short_scrape_records_first_touch_without_verifying_step():
    result = feed(GaitRecorder(weight=10.0), [
        (0.01, force(1, "first", 0.5)),
        (0.02, {}),
    ])
    event = result["events"][0]
    assert event["first_touch_label"] == "first"
    assert event["first_sustained_label"] is None
    assert event["verified"] is False
    assert event["support_intervals"]["first"] == []
    assert result["verified_steps"] == []
    assert result["candidate_flip_count"] == 0


def test_middle_support_precedes_later_endpoint_and_is_ambiguous():
    result = feed(GaitRecorder(weight=10.0), [
        (0.01, force(1, "middle", 2.0)),
        (0.02, force(1, "middle", 2.0)),
        (0.03, force(1, "first", 2.0)),
        (0.04, force(1, "first", 2.0)),
    ])
    event = result["events"][0]
    assert event["first_touch_label"] == "middle"
    assert event["middle_support_time"] == pytest.approx(0.02)
    assert event["middle_preceded"] is True
    assert event["endpoint_label"] == "first"
    assert event["ambiguous"] is True
    assert event["verified"] is True
    assert result["candidate_flip_count"] == 1
    assert result["confirmed_flip_count"] == 0


def test_later_other_endpoint_is_recorded_as_closure_without_changing_first_label():
    result = feed(GaitRecorder(weight=10.0), [
        (0.01, force(1, "first", 2.0)),
        (0.02, force(1, "first", 2.0)),
        (0.03, {1: {"first": 2.0, "last": 2.0, "middle": 0.0}}),
        (0.04, {1: {"first": 2.0, "last": 2.0, "middle": 0.0}}),
        (0.05, {1: {"first": 2.0, "last": 0.0, "middle": 0.0}}),
    ])
    event = result["events"][0]
    assert event["endpoint_label"] == "first"
    assert [support["label"] for support in event["endpoint_supports"]] == ["first", "last"]
    assert event["ambiguous"] is False
    assert result["confirmed_flip_count"] == 1


def test_simultaneous_endpoint_support_is_ambiguous():
    result = feed(GaitRecorder(weight=10.0), [
        (0.01, {1: {"first": 2.0, "last": 2.0, "middle": 0.0}}),
        (0.02, {1: {"first": 2.0, "last": 2.0, "middle": 0.0}}),
    ])
    event = result["events"][0]
    assert event["endpoint_label"] == "simultaneous"
    assert event["ambiguous"] is True
    assert result["candidate_flip_count"] == 0
    assert result["confirmed_flip_count"] == 0


def test_middle_sustained_alone_does_not_verify_a_stair():
    result = feed(GaitRecorder(weight=10.0), [
        (0.01, force(1, "middle", 2.0)),
        (0.02, force(1, "middle", 2.0)),
        (0.03, {}),
    ])
    event = result["events"][0]
    assert event["first_sustained_label"] == "middle"
    assert event["verified"] is False
    assert result["verified_steps"] == []
    assert result["candidate_flip_count"] == 0


def test_continuous_alternating_endpoint_support_is_confirmed():
    result = feed(GaitRecorder(weight=10.0), [
        (0.01, force(1, "first", 2.0)),
        (0.02, force(1, "first", 2.0)),
        (0.03, {}),
        (0.04, force(2, "last", 2.0)),
        (0.05, force(2, "last", 2.0)),
        (0.06, {}),
        (0.07, force(3, "first", 2.0)),
        (0.08, force(3, "first", 2.0)),
    ])
    assert result["verified_steps"] == [1, 2, 3]
    assert result["candidate_flip_count"] == 3
    assert result["confirmed_flip_count"] == 3
    assert result["confirmed_endpoint_sequence"] == ["first", "last", "first"]


def test_same_endpoint_sliding_stops_alternating_prefix():
    result = feed(GaitRecorder(weight=10.0), [
        (0.01, force(1, "first", 2.0)),
        (0.02, force(1, "first", 2.0)),
        (0.03, force(2, "first", 2.0)),
        (0.04, force(2, "first", 2.0)),
    ])
    assert result["verified_steps"] == [1, 2]
    assert result["candidate_flip_count"] == 1
    assert result["confirmed_flip_count"] == 1


def test_jump_does_not_advance_verified_sequence_past_missing_stair():
    result = feed(GaitRecorder(weight=10.0), [
        (0.01, force(1, "first", 2.0)),
        (0.02, force(1, "first", 2.0)),
        (0.03, force(3, "last", 2.0)),
        (0.04, force(3, "last", 2.0)),
    ])
    events = {event["step"]: event for event in result["events"]}
    assert events[1]["verified"] is True
    assert events[3]["verified"] is False
    assert "expected 2" in events[3]["verification_reason"]
    assert result["verified_steps"] == [1]
    assert result["candidate_flip_count"] == 1
    assert result["confirmed_flip_count"] == 1
