"""Offline kinematic review of endpoint overstep motion in saved gait frames."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Fixed model convention: material ``first`` is the upper/support-side end and ``last`` is the free end.
# This reviewer applies that convention; it does not infer or swap the endpoint labels from poses.
LABELS = ("first", "last")
DESTINATION_STEPS = (2, 3, 4)


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON-style object")
    return dict(value)


def _number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _xyz(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(_number(item, name) for item in value)  # type: ignore[return-value]


def _box_rotation_z(quaternion: Sequence[float]) -> tuple[float, float, float]:
    if len(quaternion) != 4:
        raise ValueError("each quaternion must be wxyz")
    w, x, y, z = (_number(item, "quaternion") for item in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # The world-z projection is the third row of the local-to-world rotation.
    return (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))


def _frame_list(frames: Any) -> list[dict[str, Any]]:
    if isinstance(frames, Mapping):
        frames = frames.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("frames must be a list or an object containing a frames list")
    result = [_mapping(frame) for frame in frames]
    if not result:
        raise ValueError("frames must not be empty")
    result.sort(key=lambda frame: _number(frame.get("time"), "frame time"))
    previous_time = -math.inf
    position_count: int | None = None
    for frame in result:
        time = _number(frame.get("time"), "frame time")
        if time <= previous_time:
            raise ValueError("frame times must be strictly increasing")
        previous_time = time
        positions = frame.get("positions")
        quaternions = frame.get("quaternions")
        if not isinstance(positions, Sequence) or not isinstance(quaternions, Sequence) or len(positions) != len(quaternions):
            raise ValueError("each frame must have matching position and quaternion arrays")
        if position_count is None:
            position_count = len(positions)
        if len(positions) != position_count or position_count < 2:
            raise ValueError("all frames must contain the same non-empty segment set")
    return result


def _support_events(diagnostics: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wrapper = dict(diagnostics)
    nested = wrapper.get("support_diagnostics")
    if isinstance(nested, Mapping):
        diag = dict(nested)
    else:
        diag = wrapper
    events = diag.get("events")
    if not isinstance(events, list):
        events = diag.get("first_three_endpoint_support_events")
    if not isinstance(events, list):
        evidence = wrapper.get("support_evidence")
        if isinstance(evidence, Mapping):
            events = evidence.get("first_three_endpoint_support_events")
    if not isinstance(events, list):
        events = []
    summary_context = wrapper.get("_summary_context")
    if not isinstance(summary_context, Mapping):
        summary_context = wrapper if "status" in wrapper or "max_penetration" in wrapper else {}
    return [_mapping(event) for event in events], dict(summary_context)


def _pivot_force_threshold(diagnostics: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float | None, str | None]:
    wrapper = dict(diagnostics)
    nested = wrapper.get("support_diagnostics")
    diag = nested if isinstance(nested, Mapping) else wrapper
    recorded = diag.get("support_threshold_n")
    if recorded is not None:
        try:
            threshold = _number(recorded, "support_threshold_n")
            if threshold >= 0.0:
                return threshold, "support_diagnostics.support_threshold_n"
        except (TypeError, ValueError):
            pass
    material = config.get("material", {})
    scene = config.get("scene", {})
    if isinstance(material, Mapping) and isinstance(scene, Mapping):
        try:
            mass = _number(material.get("mass"), "material.mass")
            gravity = _number(scene.get("gravity"), "scene.gravity")
            if mass > 0.0 and gravity >= 0.0:
                return 0.1 * mass * gravity, "fallback: 0.1 * material.mass * scene.gravity"
        except (TypeError, ValueError):
            pass
    return None, None


def _resolved_segments_per_turn(config: Mapping[str, Any], segment_count: int) -> int:
    numerics = config.get("numerics", {})
    if not isinstance(numerics, Mapping):
        numerics = {}
    resolved = numerics.get("resolved", {})
    if isinstance(resolved, Mapping) and resolved.get("segments_per_turn") is not None:
        value = int(resolved["segments_per_turn"])
    elif numerics.get("segments_per_turn") is not None:
        value = int(numerics["segments_per_turn"])
    else:
        material = config.get("material", {})
        turns = int(material.get("turns", 0)) if isinstance(material, Mapping) else 0
        value = segment_count // turns if turns > 0 and segment_count % turns == 0 else 0
    if value <= 0 or value * 2 > segment_count:
        raise ValueError("cannot resolve a valid segments_per_turn terminal group")
    return value


def _half_sizes(metadata: Mapping[str, Any], segment_count: int) -> list[tuple[float, float, float]]:
    raw = metadata.get("half_sizes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("metadata.half_sizes is required to recover box envelopes")
    if len(raw) == 3 and all(isinstance(item, (int, float)) for item in raw):
        one = _xyz(raw, "half_sizes")
        sizes = [one] * segment_count
    else:
        sizes = [_xyz(item, "half_sizes") for item in raw]
    if len(sizes) != segment_count or any(any(size <= 0.0 for size in row) for row in sizes):
        raise ValueError("metadata.half_sizes must provide positive xyz half-sizes for every segment")
    return sizes


def _segment_states(frame: Mapping[str, Any], half_sizes: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    positions = frame["positions"]
    quaternions = frame["quaternions"]
    states: list[dict[str, Any]] = []
    for position, quaternion, half in zip(positions, quaternions, half_sizes):
        center = _xyz(position, "segment position")
        rotation_z = _box_rotation_z(quaternion)
        z_extent = sum(abs(axis) * side for axis, side in zip(rotation_z, half))
        states.append({"center": center, "z_min": center[2] - z_extent, "z_max": center[2] + z_extent})
    return states


def _terminal_geometry(states: Sequence[Mapping[str, Any]], label: str, segments_per_turn: int) -> dict[str, Any]:
    count = len(states)
    begin, end = (0, segments_per_turn) if label == "first" else (count - segments_per_turn, count)
    group = states[begin:end]
    return {
        "center": tuple(sum(item["center"][axis] for item in group) / len(group) for axis in range(3)),
        "z_min": min(item["z_min"] for item in group),
        "z_max": max(item["z_max"] for item in group),
        "material_start": begin,
        "material_end_exclusive": end,
    }


def _event_for_step(events: Sequence[Mapping[str, Any]], step: int) -> Mapping[str, Any] | None:
    return next((event for event in events if int(event.get("step", -1)) == step), None)


def _label_indices(label: str, segment_count: int, segments_per_turn: int) -> range:
    if label == "first":
        return range(0, segments_per_turn)
    if label == "last":
        return range(segment_count - segments_per_turn, segment_count)
    return range(0)


def _contact_matches(frame: Mapping[str, Any], step: int, label: str, segment_count: int, segments_per_turn: int) -> list[dict[str, Any]]:
    material_indices = set(_label_indices(label, segment_count, segments_per_turn))
    matches = []
    details = frame.get("contact_details", [])
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        return matches
    for raw in details:
        if not isinstance(raw, Mapping):
            continue
        try:
            if raw.get("kind", "stair") != "stair" or int(raw.get("stair_step", -1)) != step:
                continue
            if raw.get("surface", "tread") != "tread" or int(raw.get("material_index", -1)) not in material_indices:
                continue
        except (TypeError, ValueError):
            continue
        matches.append(dict(raw))
    return matches


def _contact_force_sample(frame: Mapping[str, Any], step: int, label: str, segment_count: int, segments_per_turn: int) -> dict[str, Any]:
    rows = _contact_matches(frame, step, label, segment_count, segments_per_turn)
    forces = []
    complete = True
    for row in rows:
        try:
            force = _number(row.get("normal_force"), "contact normal_force")
            if force < 0.0:
                raise ValueError("contact normal_force must be non-negative")
            forces.append(force)
        except (TypeError, ValueError):
            complete = False
    return {
        "contact_count": len(rows),
        "force_data_complete": complete,
        "normal_force_sum_n": sum(forces) if complete else None,
    }


def _event_support_is_unambiguous(event: Mapping[str, Any], label: str, dwell_s: float) -> tuple[bool, str | None, dict[str, Any] | None]:
    if event.get("verified") is not True:
        return False, "target_step_not_verified", None
    if event.get("ambiguous") is True or event.get("ambiguity_reasons"):
        return False, "target_step_support_ambiguous", None
    if event.get("middle_preceded") is True:
        return False, "middle_support_preceded_endpoint", None
    if event.get("first_touch_label") != label or event.get("endpoint_label") != label or event.get("first_sustained_label") != label:
        return False, "target_endpoint_not_first_sustained_support", None
    supports = event.get("endpoint_supports", [])
    if not isinstance(supports, Sequence) or isinstance(supports, (str, bytes)):
        supports = []
    for support in supports:
        if isinstance(support, Mapping) and support.get("label") == label:
            duration = float(support.get("duration_s", 0.0))
            if duration + 1.0e-12 < dwell_s:
                return False, "target_endpoint_dwell_too_short", dict(support)
            return True, None, dict(support)
    return False, "target_endpoint_sustained_support_missing", None


def _sample_resolution(times: Sequence[float]) -> dict[str, Any]:
    intervals = [right - left for left, right in zip(times, times[1:])]
    if not intervals:
        return {"frame_count": len(times), "min_dt_s": None, "median_dt_s": None, "max_dt_s": None}
    ordered = sorted(intervals)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
    return {"frame_count": len(times), "min_dt_s": min(intervals), "median_dt_s": median, "max_dt_s": max(intervals)}


def _terminal_center_x_range(
    frames: Sequence[Mapping[str, Any]],
    states_by_frame: Sequence[Sequence[Mapping[str, Any]]],
    indices: Sequence[int],
    label: str,
    step: int,
    segment_count: int,
    segments_per_turn: int,
) -> dict[str, Any]:
    values = []
    for frame_index in indices:
        if _contact_matches(frames[frame_index], step, label, segment_count, segments_per_turn):
            geometry = _terminal_geometry(states_by_frame[frame_index], label, segments_per_turn)
            values.append((float(frames[frame_index]["time"]), float(geometry["center"][0])))
    if not values:
        return {"x_range_m": None, "x_drift_m": None, "sample_count": 0, "method": "no sampled terminal tread contacts"}
    xs = [item[1] for item in values]
    return {"x_range_m": [min(xs), max(xs)], "x_drift_m": max(xs) - min(xs), "sample_count": len(values), "method": "terminal group center over saved tread-contact frames", "samples": values}


def _review_one_swing(
    destination_step: int,
    frames: Sequence[Mapping[str, Any]],
    times: Sequence[float],
    states_by_frame: Sequence[Sequence[Mapping[str, Any]]],
    events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    segments_per_turn: int,
    thickness: float,
    radius: float,
    dwell_s: float,
    run_complete: bool,
    support_events_available: bool,
    pivot_force_threshold_n: float | None,
    pivot_force_threshold_source: str | None,
) -> dict[str, Any]:
    source_step = destination_step - 2
    previous_event = _event_for_step(events, destination_step - 1)
    target_event = _event_for_step(events, destination_step)
    expected_target_label = "first" if destination_step % 2 == 0 else "last"
    result: dict[str, Any] = {
        "destination_step": destination_step,
        "source_step": source_step,
        "pivot_step": destination_step - 1,
        "target_label": expected_target_label,
        "pivot_label": None,
        "status": "inconclusive",
        "passed": False,
        "reasons": [],
    }
    scene = config.get("scene", {})
    if not isinstance(scene, Mapping) or "step_height" not in scene or "step_count" not in scene or "step_depth" not in scene:
        result["status"] = "inconclusive"
        result["reasons"].append("stair_geometry_missing_from_config")
        return result
    if previous_event is None or target_event is None:
        missing_from_complete_run = run_complete and support_events_available
        result["status"] = "failed" if missing_from_complete_run else "inconclusive"
        result["reasons"].append("required_support_event_missing_from_completed_run" if missing_from_complete_run else "required_support_event_missing")
        return result

    pivot_label = previous_event.get("first_sustained_label") or previous_event.get("endpoint_label")
    result["pivot_label"] = pivot_label
    if pivot_label not in LABELS or pivot_label == expected_target_label:
        result["status"] = "failed"
        result["reasons"].append("support_endpoints_do_not_alternate_as_expected")
        return result
    previous_support_passed, previous_support_reason, _ = _event_support_is_unambiguous(previous_event, pivot_label, dwell_s)
    if not previous_support_passed:
        result["status"] = "failed" if previous_event.get("verified") is True else "inconclusive"
        result["reasons"].append(previous_support_reason or "previous_pivot_support_not_verified")
        return result
    try:
        start_touch = _number(previous_event.get("first_touch_time"), "previous first_touch_time")
        target_touch = _number(target_event.get("first_touch_time"), "target first_touch_time")
    except (TypeError, ValueError):
        result["reasons"].append("support_event_first_touch_time_missing")
        return result
    if target_touch <= start_touch:
        result["status"] = "failed"
        result["reasons"].append("support_event_times_not_increasing")
        return result

    start_index = max(0, bisect.bisect_left(times, start_touch) - 1)
    end_index = bisect.bisect_left(times, target_touch)
    if end_index >= len(frames):
        end_index = len(frames) - 1
    frame_indices = list(range(start_index, end_index + 1))
    result["window"] = {
        "previous_first_touch_s": start_touch,
        "target_first_touch_s": target_touch,
        "start_frame_index": start_index,
        "end_frame_index": end_index,
        "start_frame_time_s": times[start_index],
        "end_frame_time_s": times[end_index],
        "includes_one_prior_sample": start_index < bisect.bisect_left(times, start_touch),
        "sample_count": len(frame_indices),
    }
    if len(frame_indices) < 2:
        result["reasons"].append("too_few_samples_for_crossing")
        return result

    target_geometries = {
        frame_index: _terminal_geometry(states_by_frame[frame_index], expected_target_label, segments_per_turn)
        for frame_index in frame_indices
    }
    pivot_geometries = {
        frame_index: _terminal_geometry(states_by_frame[frame_index], pivot_label, segments_per_turn)
        for frame_index in frame_indices
    }
    relative_x = {
        frame_index: target_geometries[frame_index]["center"][0] - pivot_geometries[frame_index]["center"][0]
        for frame_index in frame_indices
    }
    crossing: tuple[int, int] | None = None
    for left, right in zip(frame_indices, frame_indices[1:]):
        if relative_x[left] < 0.0 <= relative_x[right]:
            crossing = (left, right)
            break
    start_relative = relative_x[frame_indices[0]]
    end_relative = relative_x[frame_indices[-1]]
    result["x_order_start_m"] = start_relative
    result["x_order_end_m"] = end_relative
    if crossing is None:
        if start_relative >= 0.0:
            result["status"] = "failed"
            result["reasons"].append("target_end_did_not_start_behind_support_end")
        elif end_relative <= 0.0 and target_event.get("verified") is True:
            result["status"] = "failed"
            result["reasons"].append("x_order_did_not_cross_before_landing")
        else:
            result["reasons"].append("x_crossing_not_observed_in_available_window")
        return result

    left, right = crossing
    left_relative, right_relative = relative_x[left], relative_x[right]
    fraction = -left_relative / (right_relative - left_relative)
    left_time, right_time = times[left], times[right]
    crossing_time = left_time + fraction * (right_time - left_time)
    box_clearance = min(target_geometries[left]["z_min"], target_geometries[right]["z_min"]) - max(
        pivot_geometries[left]["z_max"], pivot_geometries[right]["z_max"]
    )
    threshold = max(1.0e-4, 0.5 * thickness)
    result["x_crossing"] = {
        "bracket_s": [left_time, right_time],
        "bracket_width_s": right_time - left_time,
        "interpolated_time_s": crossing_time,
        "relative_x_before_m": left_relative,
        "relative_x_after_m": right_relative,
        "center_z_difference_interpolated_m": (
            target_geometries[left]["center"][2]
            + fraction * (target_geometries[right]["center"][2] - target_geometries[left]["center"][2])
            - pivot_geometries[left]["center"][2]
            - fraction * (pivot_geometries[right]["center"][2] - pivot_geometries[left]["center"][2])
        ),
        "conservative_box_clearance_m": box_clearance,
        "clearance_threshold_m": threshold,
        "clearance_passed": box_clearance >= threshold,
        "time_uncertainty_bound_s": right_time - left_time,
    }

    pivot_force_samples = []
    for frame_index in crossing:
        sample = _contact_force_sample(
            frames[frame_index], destination_step - 1, pivot_label,
            len(states_by_frame[frame_index]), segments_per_turn,
        )
        pivot_force_samples.append({"frame_index": frame_index, "time_s": times[frame_index], **sample})
    if pivot_force_threshold_n is None:
        pivot_force_status = "inconclusive"
        pivot_force_passed = False
        pivot_force_reason = "pivot_support_force_threshold_unavailable"
    elif any(sample["force_data_complete"] and sample["normal_force_sum_n"] >= pivot_force_threshold_n for sample in pivot_force_samples):
        pivot_force_status = "passed"
        pivot_force_passed = True
        pivot_force_reason = None
    elif all(sample["force_data_complete"] for sample in pivot_force_samples):
        pivot_force_status = "failed"
        pivot_force_passed = False
        pivot_force_reason = "pivot_terminal_tread_force_below_threshold_at_both_crossing_samples"
    else:
        pivot_force_status = "inconclusive"
        pivot_force_passed = False
        pivot_force_reason = "pivot_terminal_tread_force_missing_at_crossing_samples"
    result["pivot_support_at_crossing"] = {
        "step": destination_step - 1,
        "label": pivot_label,
        "threshold_n": pivot_force_threshold_n,
        "threshold_source": pivot_force_threshold_source,
        "samples": pivot_force_samples,
        "status": pivot_force_status,
        "passed": pivot_force_passed,
        "interpretation": "sampled force evidence at either saved x-crossing bracket frame; not a continuous-force assertion",
    }
    if pivot_force_reason:
        result["reasons"].append(pivot_force_reason)

    top_z = _number(scene["step_height"], "step_height") * (int(scene["step_count"]) - source_step)
    depth = _number(scene["step_depth"], "step_depth")
    source_x_range = [depth * source_step, depth * (source_step + 1)]
    source_evidence_frames = []
    source_contact_frames = []
    source_near_plane_frames = []
    lift_samples = []
    for frame_index in frame_indices:
        if times[frame_index] > left_time + 1.0e-12:
            break
        geometry = target_geometries[frame_index]
        center = geometry["center"]
        contact_rows = _contact_matches(frames[frame_index], source_step, expected_target_label, len(states_by_frame[frame_index]), segments_per_turn)
        near_x = source_x_range[0] - radius <= center[0] <= source_x_range[1] + radius
        near_z = abs(geometry["z_min"] - top_z) <= threshold
        evidence_kinds = []
        if contact_rows:
            source_contact_frames.append(frame_index)
            evidence_kinds.append("terminal_tread_contact")
        if near_x and near_z:
            source_near_plane_frames.append(frame_index)
            evidence_kinds.append("terminal_box_near_source_plane")
        if evidence_kinds:
            source_evidence_frames.append({"frame_index": frame_index, "time_s": times[frame_index], "evidence": evidence_kinds})
        if source_evidence_frames and frame_index >= source_evidence_frames[0]["frame_index"]:
            lift_samples.append((times[frame_index], geometry["z_min"] - top_z))
    source_evidence = bool(source_evidence_frames)
    lift_max = max((item[1] for item in lift_samples), default=None)
    lift_time = max(lift_samples, key=lambda item: item[1])[0] if lift_samples else None
    lift_passed = source_evidence and lift_max is not None and lift_max >= threshold
    result["source_tread"] = {
        "step": source_step,
        "top_z_m": top_z,
        "x_range_m": source_x_range,
        "contact_observed": bool(source_contact_frames),
        "near_plane_observed": bool(source_near_plane_frames),
        "evidence_frames": source_evidence_frames,
    }
    result["lift"] = {
        "maximum_lower_box_clearance_above_source_m": lift_max,
        "maximum_time_s": lift_time,
        "threshold_m": threshold,
        "passed": bool(lift_passed),
        "measured_before_x_crossing": bool(lift_time is not None and lift_time <= left_time),
        "sample_count": len(lift_samples),
    }
    if not source_evidence:
        result["reasons"].append("no_prior_source_tread_contact_or_near_plane_evidence")
    elif not lift_passed:
        result["reasons"].append("terminal_end_did_not_clear_source_tread_by_threshold_before_crossing")
    if box_clearance < threshold:
        result["reasons"].append("terminal_boxes_not_separated_above_support_end_at_x_crossing")

    support_passed, support_reason, support_record = _event_support_is_unambiguous(target_event, expected_target_label, dwell_s)
    result["target_support"] = {
        "step": destination_step,
        "first_touch_time_s": target_touch,
        "first_touch_label": target_event.get("first_touch_label"),
        "sustained_time_s": target_event.get("first_sustained_time"),
        "sustained_label": target_event.get("first_sustained_label"),
        "endpoint_label": target_event.get("endpoint_label"),
        "verified": target_event.get("verified") is True,
        "ambiguous": target_event.get("ambiguous", False),
        "middle_preceded": target_event.get("middle_preceded", False),
        "endpoint_support": support_record,
        "unambiguous_sustained_endpoint_support": support_passed,
    }
    if support_reason:
        result["reasons"].append(support_reason)

    target_top = _number(scene["step_height"], "step_height") * (int(scene["step_count"]) - destination_step)
    destination_x_range = [depth * destination_step, depth * (destination_step + 1)]
    sustained_time = target_event.get("first_sustained_time")
    landing_window_start = target_touch - max((times[i + 1] - times[i] for i in range(len(times) - 1)), default=0.0)
    landing_window_end = (_number(sustained_time, "first_sustained_time") if sustained_time is not None else target_touch) + max(
        (times[i + 1] - times[i] for i in range(len(times) - 1)), default=0.0
    )
    landing_samples = []
    target_contacts = []
    for frame_index, time in enumerate(times):
        if time < landing_window_start or time > landing_window_end or time < crossing_time:
            continue
        geometry = _terminal_geometry(states_by_frame[frame_index], expected_target_label, segments_per_turn)
        center = geometry["center"]
        in_x = destination_x_range[0] - radius <= center[0] <= destination_x_range[1] + radius
        on_z = abs(geometry["z_min"] - target_top) <= threshold
        contacts = _contact_matches(frames[frame_index], destination_step, expected_target_label, len(states_by_frame[frame_index]), segments_per_turn)
        if contacts:
            target_contacts.extend({"frame_index": frame_index, "time_s": time, "material_index": contact.get("material_index"), "position": contact.get("position")} for contact in contacts)
        if in_x and on_z:
            landing_samples.append({"frame_index": frame_index, "time_s": time, "center_xyz_m": list(center), "z_min_m": geometry["z_min"], "z_max_m": geometry["z_max"], "x_in_tread": in_x, "bottom_on_tread": on_z, "terminal_tread_contact_observed": bool(contacts)})
    landing_geometry_passed = bool(landing_samples)
    result["landing"] = {
        "target_top_z_m": target_top,
        "target_x_range_m": destination_x_range,
        "search_window_s": [landing_window_start, landing_window_end],
        "geometry_observed": landing_geometry_passed,
        "samples": landing_samples,
        "terminal_tread_contact_details": target_contacts,
        "contact_detail_observed": bool(target_contacts),
        "support_event_confirms_tread_force": support_passed,
    }
    if not landing_geometry_passed:
        result["reasons"].append("target_terminal_box_not_sampled_on_destination_tread")

    pivot_contact_indices = [
        index for index in frame_indices
        if _contact_matches(frames[index], destination_step - 1, pivot_label, len(states_by_frame[index]), segments_per_turn)
    ]
    result["support_end_x_drift"] = _terminal_center_x_range(
        frames, states_by_frame, pivot_contact_indices, pivot_label, destination_step - 1, len(states_by_frame[0]), segments_per_turn
    )

    other_criteria = [source_evidence, bool(lift_passed), box_clearance >= threshold, support_passed, landing_geometry_passed]
    known_failure = any(not criterion for criterion in other_criteria)
    if not source_evidence or (pivot_force_status == "inconclusive" and not known_failure):
        result["status"] = "inconclusive"
    elif known_failure or not pivot_force_passed:
        result["status"] = "failed"
    else:
        result["status"] = "passed"
        result["passed"] = True
    return result


def review_gait_motion(
    frames: Any,
    metadata: Any,
    config: Any,
    support_diagnostics: Any,
) -> dict[str, Any]:
    """Review three post-settle endpoint swings using saved box poses and support records.

    The result is a kinematic motion assessment only. Initial-condition, mesh,
    timestep, perturbation, and experimental-validation decisions remain separate.
    """
    frame_rows = _frame_list(frames)
    metadata_row = _mapping(metadata)
    config_row = _mapping(config)
    diagnostics_row = _mapping(support_diagnostics)
    events, summary_context = _support_events(diagnostics_row)
    times = [_number(frame["time"], "frame time") for frame in frame_rows]
    segment_count = len(frame_rows[0]["positions"])
    declared_count = int(metadata_row.get("segment_count", segment_count))
    if declared_count != segment_count:
        raise ValueError("metadata segment_count does not match saved frame geometry")
    sizes = _half_sizes(metadata_row, segment_count)
    segments_per_turn = _resolved_segments_per_turn(config_row, segment_count)
    material = config_row.get("material", {})
    if not isinstance(material, Mapping):
        material = {}
    numerics = config_row.get("numerics", {})
    if not isinstance(numerics, Mapping):
        numerics = {}
    thickness = _number(material.get("strip_thickness", 2.0 * sizes[0][2]), "strip_thickness")
    radius = _number(material.get("radius", 0.0), "radius")
    nested_diagnostics = diagnostics_row.get("support_diagnostics")
    nested_dwell = nested_diagnostics.get("dwell_s", 0.02) if isinstance(nested_diagnostics, Mapping) else 0.02
    dwell_s = _number(diagnostics_row.get("dwell_s", nested_dwell), "dwell_s")
    pivot_force_threshold_n, pivot_force_threshold_source = _pivot_force_threshold(diagnostics_row, config_row)
    states_by_frame = [_segment_states(frame, sizes) for frame in frame_rows]
    resolution = _sample_resolution(times)
    duration = numerics.get("duration")
    max_dt = resolution.get("max_dt_s") or 0.0
    run_complete = duration is not None and times[-1] >= float(duration) - 1.5 * max_dt
    swings = [
        _review_one_swing(step, frame_rows, times, states_by_frame, events, config_row, segments_per_turn, thickness, radius, dwell_s, run_complete, bool(events), pivot_force_threshold_n, pivot_force_threshold_source)
        for step in DESTINATION_STEPS
    ]
    completed = sum(1 for swing in swings if swing["passed"])
    statuses = [swing["status"] for swing in swings]
    if completed == len(DESTINATION_STEPS):
        status = "passed"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "inconclusive"
    initial_diagnostics = metadata_row.get("initial_pose_diagnostics", {})
    if not isinstance(initial_diagnostics, Mapping):
        initial_diagnostics = {}
    frame_penetration = [
        float(frame.get("metrics", {}).get("max_penetration"))
        for frame in frame_rows
        if isinstance(frame.get("metrics"), Mapping) and frame["metrics"].get("max_penetration") is not None
    ]
    initial_context: dict[str, Any] = {
        "scope": "motion-only; full numerical and initial-condition acceptance is not assessed",
        "initial_max_penetration_m": initial_diagnostics.get("initial_max_penetration_m"),
        "initial_clearances_m": initial_diagnostics.get("target_clearances_m"),
        "max_penetration_in_saved_frame_metrics_m": max(frame_penetration) if frame_penetration else summary_context.get("max_penetration"),
        "summary_status": summary_context.get("validation_status", summary_context.get("status")),
    }
    return {
        "schema": "slinky-lab.gait-motion-review.v1",
        "acceptance_scope": "kinematic_motion_only",
        "whole_case_acceptance": "not_assessed",
        "endpoint_order_assumption": {
            "first": "upper/support-side terminal end",
            "last": "free/moving terminal end",
            "source": "fixed model convention; endpoint roles are not inferred from saved poses",
        },
        "validation_limits": {
            "initial_condition_acceptance": "not assessed by the motion-only checker",
            "numerical_mesh_and_timestep_convergence": "not assessed by the motion-only checker",
            "experimental_si_calibration": "not assessed by the motion-only checker",
        },
        "status": status,
        "passed": status == "passed",
        "completed_swing_count": completed,
        "required_swing_count": len(DESTINATION_STEPS),
        "analyzed_swing_count": len(swings),
        "sampling": resolution,
        "saved_run_reaches_configured_duration": run_complete,
        "segments_per_turn": segments_per_turn,
        "terminal_group_size_segments": segments_per_turn,
        "clearance_and_lift_threshold_m": max(1.0e-4, 0.5 * thickness),
        "initial_condition_context": initial_context,
        "swing_windows": swings,
        "uncertainty": {
            "x_crossing_time": "bounded by each adjacent saved-frame bracket; linear interpolation is descriptive only",
            "box_clearance": "envelope separation evaluated at the two saved crossing-bracket frames; conservative across those samples only, not a bound on unsampled inter-frame geometry",
            "lift": "maximum over saved frames only; sub-frame lift peaks may be missed",
            "support_drift": "saved terminal-group centers on saved source-tread-contact frames; not a frictional slip estimate",
        },
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _case_input(case_dir: Path, name: str, required: bool = True) -> Any:
    path = case_dir / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required input not found: {path}")
        return None
    return _read_json(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="case directory containing JSON frames, metadata, config, and summary")
    parser.add_argument("--output", type=Path, default=Path("-"), help="output JSON path, or '-' for stdout")
    parser.add_argument("--frames", type=Path, help="override frames.json path")
    parser.add_argument("--metadata", type=Path, help="override metadata.json path")
    parser.add_argument("--config", type=Path, help="override config.json path")
    parser.add_argument("--summary", type=Path, help="override summary.json path")
    args = parser.parse_args(argv)
    case_dir = args.case_dir
    frames = _read_json(args.frames or case_dir / "frames.json")
    metadata = _read_json(args.metadata or case_dir / "metadata.json")
    config = _read_json(args.config or case_dir / "config.json")
    summary = _read_json(args.summary or case_dir / "summary.json")
    diagnostics = _case_input(case_dir, "support-diagnostics.json", required=False)
    if diagnostics is None:
        diagnostics = summary.get("support_diagnostics", summary)
    if isinstance(diagnostics, Mapping):
        diagnostics = dict(diagnostics)
        diagnostics["_summary_context"] = summary
    result = review_gait_motion(frames, metadata, config, diagnostics)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if str(args.output) == "-":
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(f"gait motion: {result['status']}; completed={result['completed_swing_count']}/{result['required_swing_count']}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
