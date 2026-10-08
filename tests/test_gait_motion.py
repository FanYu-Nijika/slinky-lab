from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("review_gait_motion", SCRIPT_DIR / "review_gait_motion.py")
review = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(review)


def _case(*, omit_step4: bool = False, airborne_first: bool = False, same_end_order: bool = False, airborne_pivot: bool = False):
    half_sizes = [[0.01, 0.004, 0.00225]] * 4
    metadata = {"segment_count": 4, "half_sizes": half_sizes, "initial_pose_diagnostics": {"initial_max_penetration_m": 0.0}}
    config = {
        "material": {"turns": 2, "radius": 0.01, "strip_thickness": 0.0045, "mass": 0.05},
        "scene": {"step_height": 0.1, "step_depth": 0.1, "step_count": 6, "gravity": 10.0},
        "numerics": {"resolved": {"segments_per_turn": 2}, "duration": 3.0},
    }
    events = []
    labels = {1: "last", 2: "first", 3: "last", 4: "first"}
    touches = {1: 0.1, 2: 0.9, 3: 1.7, 4: 2.5}
    for step in range(1, 5):
        if omit_step4 and step == 4:
            continue
        label = labels[step]
        touch = touches[step]
        events.append({
            "step": step,
            "first_touch_time": touch,
            "first_touch_label": label,
            "first_sustained_time": touch + 0.03,
            "first_sustained_label": label,
            "endpoint_label": label,
            "endpoint_supports": [{"label": label, "duration_s": 0.04}],
            "verified": True,
            "ambiguous": False,
            "ambiguity_reasons": [],
            "middle_preceded": False,
        })
    diagnostics = {"dwell_s": 0.02, "support_threshold_n": 0.5, "events": events}

    frames = []
    for sample in range(31):
        time = sample / 10.0

        # Each swing has an explicit lift, an x-order reversal above the pivot,
        # then a descent onto the next lower tread.
        first_x, first_bottom = 0.05, 0.6
        if time <= 0.1:
            first_x, first_bottom = 0.05, 0.62 if airborne_first else 0.6
        elif time <= 0.9:
            first_states = {
                0.2: (0.08, 0.61), 0.3: (0.12, 0.62), 0.4: (0.18, 0.62),
                0.5: (0.21, 0.58), 0.6: (0.25, 0.52), 0.7: (0.25, 0.46),
                0.8: (0.25, 0.4), 0.9: (0.25, 0.4),
            }
            first_x, first_bottom = first_states[round(time, 1)]
        elif time < 1.8:
            first_x, first_bottom = 0.25, 0.4
        elif time < 2.5:
            first_states = {
                1.8: (0.29, 0.41), 1.9: (0.33, 0.42), 2.0: (0.38, 0.42),
                2.1: (0.43, 0.39), 2.2: (0.45, 0.35), 2.3: (0.45, 0.28),
                2.4: (0.45, 0.2),
            }
            first_x, first_bottom = first_states[round(time, 1)]
        else:
            first_x, first_bottom = 0.45, 0.2

        last_x, last_bottom = 0.15, 0.5
        if 0.9 < time < 1.7:
            last_states = {
                1.0: (0.19, 0.51), 1.1: (0.23, 0.52), 1.2: (0.28, 0.52),
                1.3: (0.33, 0.50), 1.4: (0.35, 0.43), 1.5: (0.35, 0.36),
                1.6: (0.35, 0.3),
            }
            last_x, last_bottom = last_states[round(time, 1)]
        elif time >= 1.7:
            last_x, last_bottom = 0.35, 0.3

        if same_end_order and time <= 0.9:
            # The moving first end remains behind the pivot through touchdown.
            first_x = min(first_x, 0.14)

        if airborne_first and time <= 0.1:
            first_bottom = 0.62

        first_center_z = first_bottom + half_sizes[0][2]
        last_center_z = last_bottom + half_sizes[0][2]
        positions = [[first_x, 0.0, first_center_z]] * 2 + [[last_x, 0.0, last_center_z]] * 2
        quaternions = [[1.0, 0.0, 0.0, 0.0]] * 4
        contacts = []
        if time <= 0.1 and not airborne_first:
            contacts.append({"kind": "stair", "surface": "tread", "stair_step": 0, "material_index": 0, "normal_force": 0.6})
        if time <= 0.9 and not (airborne_pivot and round(time, 1) in {0.3, 0.4}):
            contacts.append({"kind": "stair", "surface": "tread", "stair_step": 1, "material_index": 2, "normal_force": 0.6})
        if 0.8 <= time < 1.8:
            contacts.append({"kind": "stair", "surface": "tread", "stair_step": 2, "material_index": 0, "normal_force": 0.6})
        if 1.6 <= time < 2.5:
            contacts.append({"kind": "stair", "surface": "tread", "stair_step": 3, "material_index": 2, "normal_force": 0.6})
        if time >= 2.4:
            contacts.append({"kind": "stair", "surface": "tread", "stair_step": 4, "material_index": 0, "normal_force": 0.6})
        frames.append({"time": time, "positions": positions, "quaternions": quaternions, "contact_details": contacts})
    return frames, metadata, config, diagnostics


def test_clear_endpoint_lift_cross_and_landing_completes_three_swings():
    result = review.review_gait_motion(*_case())

    assert result["passed"]
    assert result["completed_swing_count"] == 3
    assert [item["destination_step"] for item in result["swing_windows"]] == [2, 3, 4]
    assert all(item["x_crossing"]["clearance_passed"] for item in result["swing_windows"])
    assert all(item["lift"]["passed"] for item in result["swing_windows"])
    assert all(item["landing"]["geometry_observed"] for item in result["swing_windows"])
    assert all(item["target_support"]["unambiguous_sustained_endpoint_support"] for item in result["swing_windows"])
    assert all(item["support_end_x_drift"]["sample_count"] > 0 for item in result["swing_windows"])
    assert all(item["pivot_support_at_crossing"]["passed"] for item in result["swing_windows"])
    assert all(item["pivot_support_at_crossing"]["threshold_source"] == "support_diagnostics.support_threshold_n" for item in result["swing_windows"])


def test_window_starts_at_previous_first_touch_when_sustained_support_is_later():
    frames, metadata, config, diagnostics = _case()
    diagnostics["events"][1]["first_sustained_time"] = 1.3

    result = review.review_gait_motion(frames, metadata, config, diagnostics)
    third_step = result["swing_windows"][1]

    assert third_step["x_crossing"]["interpolated_time_s"] < 1.3
    assert third_step["window"]["previous_first_touch_s"] == pytest.approx(0.9)
    assert third_step["window"]["start_frame_time_s"] == pytest.approx(0.8)


def test_same_end_order_without_x_crossover_does_not_count_as_a_swing():
    result = review.review_gait_motion(*_case(same_end_order=True))

    assert not result["passed"]
    assert result["swing_windows"][0]["status"] == "failed"
    assert "x_order_did_not_cross_before_landing" in result["swing_windows"][0]["reasons"]


def test_airborne_drop_without_source_support_evidence_is_inconclusive():
    result = review.review_gait_motion(*_case(airborne_first=True))

    first = result["swing_windows"][0]
    assert not first["passed"]
    assert first["status"] == "inconclusive"
    assert "no_prior_source_tread_contact_or_near_plane_evidence" in first["reasons"]


def test_airborne_pivot_does_not_count_despite_later_sustained_support_event():
    result = review.review_gait_motion(*_case(airborne_pivot=True))

    first = result["swing_windows"][0]
    assert not first["passed"]
    assert first["status"] == "failed"
    assert first["pivot_support_at_crossing"]["status"] == "failed"
    assert [sample["normal_force_sum_n"] for sample in first["pivot_support_at_crossing"]["samples"]] == [0.0, 0.0]
    assert "pivot_terminal_tread_force_below_threshold_at_both_crossing_samples" in first["reasons"]


def test_pivot_force_threshold_falls_back_to_ten_percent_of_weight():
    frames, metadata, config, diagnostics = _case()
    diagnostics.pop("support_threshold_n")

    result = review.review_gait_motion(frames, metadata, config, diagnostics)
    pivot = result["swing_windows"][0]["pivot_support_at_crossing"]

    assert pivot["passed"]
    assert pivot["threshold_n"] == pytest.approx(0.05)
    assert pivot["threshold_source"] == "fallback: 0.1 * material.mass * scene.gravity"


def test_pivot_force_without_saved_normal_force_is_inconclusive():
    frames, metadata, config, diagnostics = _case()
    for frame in frames:
        if frame["time"] in {0.3, 0.4}:
            for contact in frame["contact_details"]:
                if contact["stair_step"] == 1:
                    contact.pop("normal_force")

    result = review.review_gait_motion(frames, metadata, config, diagnostics)
    first = result["swing_windows"][0]

    assert first["status"] == "inconclusive"
    assert first["pivot_support_at_crossing"]["status"] == "inconclusive"
    assert "pivot_terminal_tread_force_missing_at_crossing_samples" in first["reasons"]


def test_missing_stair_geometry_is_reported_before_swing_geometry_access():
    frames, metadata, config, diagnostics = _case()
    config["scene"] = {}

    result = review.review_gait_motion(frames, metadata, config, diagnostics)

    assert result["status"] == "inconclusive"
    assert all("stair_geometry_missing_from_config" in swing["reasons"] for swing in result["swing_windows"])


def test_only_two_swings_in_full_duration_run_cannot_pass():
    result = review.review_gait_motion(*_case(omit_step4=True))

    assert result["saved_run_reaches_configured_duration"]
    assert result["completed_swing_count"] == 2
    assert not result["passed"]
    assert result["swing_windows"][2]["status"] == "failed"
    assert "required_support_event_missing_from_completed_run" in result["swing_windows"][2]["reasons"]


def test_rotated_box_vertical_extent_uses_world_z_row():
    half = [0.1, 0.01, 0.005]
    quaternion = [2**-0.5, 0.0, 2**-0.5, 0.0]
    state = review._segment_states({"positions": [[0.0, 0.0, 1.0]], "quaternions": [quaternion]}, [half])[0]

    assert state["z_min"] == pytest.approx(0.9)
