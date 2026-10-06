"""Auditable stair-support and endpoint alternation diagnostics.

The physics adapter can feed this recorder one real solver substep at a time.
It deliberately separates a first touch from sustained support: a contact that
lasts for less than ``dwell_s`` is retained as evidence, but cannot advance the
verified stair sequence.  Endpoint groups are supplied by the caller and are
expected to represent one complete terminal revolution, rather than an
arbitrary percentage of cable elements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


ENDPOINT_LABELS = ("first", "last")
SUPPORT_LABELS = ("first", "last", "middle")
SIMULTANEOUS_WINDOW_S = 0.001


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass
class _LabelRun:
    """The currently contiguous above-threshold run for one force label."""

    start_time: float | None = None
    duration_s: float = 0.0
    peak_force_n: float = 0.0
    intervals: list[dict[str, float]] = field(default_factory=list)

    def reset(self, end_time: float) -> None:
        if self.start_time is not None and self.duration_s > 0.0:
            self.intervals.append({
                "start_time": self.start_time,
                "end_time": end_time,
                "duration_s": self.duration_s,
                "peak_force_n": self.peak_force_n,
            })
        self.start_time = None
        self.duration_s = 0.0
        self.peak_force_n = 0.0


@dataclass
class _StepState:
    step: int
    first_touch_time: float | None = None
    first_touch_label: str | None = None
    first_touch_force_n: float | None = None
    first_sustained_time: float | None = None
    first_sustained_label: str | None = None
    middle_support_time: float | None = None
    middle_support_peak_force_n: float = 0.0
    middle_preceded: bool = False
    endpoint_supports: dict[str, dict[str, Any]] = field(default_factory=dict)
    ambiguous: bool = False
    ambiguity_reasons: list[str] = field(default_factory=list)
    verified: bool = False
    verified_time: float | None = None
    verification_reason: str | None = None
    verification_attempted: bool = False
    transitions: list[dict[str, Any]] = field(default_factory=list)
    runs: dict[str, _LabelRun] = field(default_factory=lambda: {label: _LabelRun() for label in SUPPORT_LABELS})

    def add_reason(self, reason: str) -> None:
        if reason not in self.ambiguity_reasons:
            self.ambiguity_reasons.append(reason)
        self.ambiguous = True


class GaitRecorder:
    """Record sustained stair support and conservative flip candidates.

    Parameters
    ----------
    weight:
        Weight force in newtons, i.e. ``M * g``.  It is used to compute the
        absolute support threshold, avoiding a hidden gravity assumption.
    support_fraction:
        Each supplied force group must reach this fraction of ``weight`` to be
        considered supporting.  The default is the requested ``0.1 * M g``.
    dwell_s:
        Required uninterrupted time above threshold.  The default is 0.02 s.
    """

    def __init__(self, weight: float, *, support_fraction: float = 0.1, dwell_s: float = 0.02, simultaneous_window_s: float = SIMULTANEOUS_WINDOW_S):
        self.weight_n = _finite(weight, "weight")
        self.support_fraction = _finite(support_fraction, "support_fraction")
        self.dwell_s = _finite(dwell_s, "dwell_s")
        self.simultaneous_window_s = _finite(simultaneous_window_s, "simultaneous_window_s")
        if self.weight_n <= 0.0:
            raise ValueError("weight must be positive")
        if self.support_fraction <= 0.0:
            raise ValueError("support_fraction must be positive")
        if self.dwell_s <= 0.0:
            raise ValueError("dwell_s must be positive")
        if self.simultaneous_window_s <= 0.0:
            raise ValueError("simultaneous_window_s must be positive")
        self.support_threshold_n = self.weight_n * self.support_fraction
        self._events: dict[int, _StepState] = {}
        self._last_time: float | None = None
        self._next_verified_step = 1

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return a current JSON-ready event snapshot ordered by stair index."""

        return [self._event_dict(self._events[step]) for step in sorted(self._events)]

    def update(self, time: float, dt: float, forces: Mapping[int | str, Mapping[str, float]] | None) -> list[dict[str, Any]]:
        """Consume one physical substep and return state transitions.

        ``forces`` maps a stair index to total tread normal force grouped by
        ``first``, ``last`` and ``middle``.  Missing groups mean zero force for
        this substep, which closes any active above-threshold run.  ``time`` is
        the end time of the substep and ``dt`` is its physical duration.
        """

        current_time = _finite(time, "time")
        step_dt = _finite(dt, "dt")
        if step_dt <= 0.0:
            raise ValueError("dt must be positive")
        if self._last_time is not None and current_time <= self._last_time:
            raise ValueError("time must increase for each substep")
        normalized = self._normalize_forces(forces)
        transitions: list[dict[str, Any]] = []
        all_steps = sorted(set(self._events).union(normalized))
        for step in all_steps:
            state = self._events.get(step)
            values = normalized.get(step, {label: 0.0 for label in SUPPORT_LABELS})
            if state is None:
                if not any(force > 0.0 for force in values.values()):
                    continue
                state = _StepState(step=step)
                self._events[step] = state
            transitions.extend(self._update_step(state, current_time, step_dt, values))
        self._last_time = current_time
        return transitions

    def finalize(self) -> dict[str, Any]:
        """Close open support intervals and return the complete audit record."""

        end_time = self._last_time if self._last_time is not None else 0.0
        for state in self._events.values():
            for run in state.runs.values():
                if run.start_time is not None:
                    run.reset(end_time)
            self._update_support_durations(state)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        """Return event records and conservative candidate/confirmed counts."""

        events = self.events
        candidate_count, candidate_sequence = self._flip_prefix(events, require_unambiguous=False)
        confirmed_count, confirmed_sequence = self._flip_prefix(events, require_unambiguous=True)
        return {
            "weight_n": self.weight_n,
            "support_fraction": self.support_fraction,
            "support_threshold_n": self.support_threshold_n,
            "dwell_s": self.dwell_s,
            "simultaneous_window_s": self.simultaneous_window_s,
            "last_time": self._last_time,
            "events": events,
            "verified_steps": [event["step"] for event in events if event["verified"]],
            "candidate_flip_count": candidate_count,
            "candidate_endpoint_sequence": candidate_sequence,
            "confirmed_flip_count": confirmed_count,
            "confirmed_endpoint_sequence": confirmed_sequence,
        }

    def _normalize_forces(self, forces: Mapping[int | str, Mapping[str, float]] | None) -> dict[int, dict[str, float]]:
        if forces is None:
            return {}
        if not isinstance(forces, Mapping):
            raise ValueError("forces must map stair indices to force groups")
        normalized: dict[int, dict[str, float]] = {}
        for raw_step, raw_values in forces.items():
            if isinstance(raw_step, bool):
                raise ValueError("stair index must be an integer")
            try:
                step = int(raw_step)
            except (TypeError, ValueError) as exc:
                raise ValueError("stair index must be an integer") from exc
            if step < 1:
                raise ValueError("stair index must be positive")
            if not isinstance(raw_values, Mapping):
                raise ValueError(f"forces[{step}] must contain first/last/middle groups")
            unknown = set(raw_values) - set(SUPPORT_LABELS)
            if unknown:
                raise ValueError(f"forces[{step}] contains unknown labels: {sorted(unknown)}")
            values = {label: _finite(raw_values.get(label, 0.0), f"forces[{step}][{label}]") for label in SUPPORT_LABELS}
            if any(value < 0.0 for value in values.values()):
                raise ValueError(f"forces[{step}] cannot be negative")
            normalized[step] = values
        return normalized

    def _update_step(self, state: _StepState, time: float, dt: float, values: Mapping[str, float]) -> list[dict[str, Any]]:
        transitions: list[dict[str, Any]] = []
        if state.first_touch_time is None and any(values[label] > 0.0 for label in SUPPORT_LABELS):
            label = max(SUPPORT_LABELS, key=lambda item: (values[item], -SUPPORT_LABELS.index(item)))
            state.first_touch_time = time
            state.first_touch_label = label
            state.first_touch_force_n = values[label]
            transition = {"time": time, "state": "first_touch", "label": label, "reason": "positive tread force observed; dwell is not yet satisfied"}
            state.transitions.append(transition)
            transitions.append({"step": state.step, **transition})
        for label in SUPPORT_LABELS:
            run = state.runs[label]
            force = values[label]
            if force >= self.support_threshold_n:
                if run.start_time is None:
                    run.start_time = time - dt
                    run.duration_s = 0.0
                    run.peak_force_n = 0.0
                run.duration_s += dt
                run.peak_force_n = max(run.peak_force_n, force)
                if run.duration_s + 1.0e-12 >= self.dwell_s:
                    if label == "middle" and state.middle_support_time is None:
                        state.middle_support_time = time
                        state.middle_support_peak_force_n = run.peak_force_n
                        transition = {"time": time, "state": "middle_sustained", "force_n": run.peak_force_n, "reason": "middle tread force reached threshold for dwell"}
                        state.transitions.append(transition)
                        transitions.append({"step": state.step, **transition})
                    if label in ENDPOINT_LABELS and label not in state.endpoint_supports:
                        state.endpoint_supports[label] = {
                            "label": label,
                            "time": time,
                            "start_time": run.start_time,
                            "duration_s": run.duration_s,
                            "peak_force_n": run.peak_force_n,
                        }
                        transition = {"time": time, "state": "endpoint_sustained", "label": label, "force_n": run.peak_force_n, "reason": "complete terminal-coil force reached threshold for dwell"}
                        state.transitions.append(transition)
                        transitions.append({"step": state.step, **transition})
                    self._update_support_durations(state)
                    if state.first_sustained_time is None:
                        state.first_sustained_time = time
                        state.first_sustained_label = label
                        if label == "middle":
                            state.add_reason("middle support was the first sustained support")
                            state.verification_reason = "middle support is recorded but does not advance verified sequence; waiting for endpoint dwell"
                            waiting = {"time": time, "state": "not_verified", "reason": state.verification_reason}
                            state.transitions.append(waiting)
                            transitions.append({"step": state.step, **waiting})
                    # A middle run can be the first sustained support, but only
                    # a later endpoint dwell may verify that stair.  This
                    # branch intentionally leaves the middle-first event
                    # ambiguous instead of treating its first support as a
                    # successful gait transition.
                    if label in ENDPOINT_LABELS and not state.verified and not state.verification_attempted and state.first_sustained_label == "middle":
                        transitions.extend(self._verify_if_contiguous(state, time))
                    elif label in ENDPOINT_LABELS and state.first_sustained_label == label and not state.verified and not state.verification_attempted:
                        transitions.extend(self._verify_if_contiguous(state, time))
            else:
                # Force values describe the interval ending at ``time``.  A
                # below-threshold sample therefore closes the previous run at
                # the beginning of this substep, rather than at its end.
                run.reset(time - dt)
        self._update_ambiguity(state)
        return transitions

    def _update_support_durations(self, state: _StepState) -> None:
        for label, support in state.endpoint_supports.items():
            run = state.runs[label]
            duration = run.duration_s if run.start_time is not None else support["duration_s"]
            support["duration_s"] = max(float(support["duration_s"]), duration)
            support["peak_force_n"] = max(float(support["peak_force_n"]), run.peak_force_n)
        if state.middle_support_time is not None:
            run = state.runs["middle"]
            state.middle_support_peak_force_n = max(state.middle_support_peak_force_n, run.peak_force_n)

    def _update_ambiguity(self, state: _StepState) -> None:
        if state.middle_support_time is not None and state.endpoint_supports:
            endpoint_time = min(float(support["time"]) for support in state.endpoint_supports.values())
            if state.middle_support_time < endpoint_time:
                state.middle_preceded = True
                state.add_reason("sustained middle support preceded sustained endpoint support")
            elif state.middle_support_time == endpoint_time:
                state.add_reason("middle and endpoint support became sustained in the same substep")
        if len(state.endpoint_supports) > 1:
            ordered = sorted(state.endpoint_supports.values(), key=lambda support: float(support["time"]))
            if float(ordered[1]["time"]) - float(ordered[0]["time"]) <= self.simultaneous_window_s:
                state.add_reason("both endpoint groups first achieved sustained support within the simultaneous window")
        if state.middle_support_time is not None and not state.endpoint_supports:
            state.add_reason("sustained middle support has no later endpoint support")

    def _verify_if_contiguous(self, state: _StepState, time: float) -> list[dict[str, Any]]:
        if state.verified:
            return []
        if state.verification_attempted:
            return []
        state.verification_attempted = True
        if state.step != self._next_verified_step:
            state.verification_reason = f"step {state.step} is not the next contiguous stair; expected {self._next_verified_step}"
            state.transitions.append({"time": time, "state": "not_verified", "reason": state.verification_reason})
            return [{"step": state.step, "time": time, "state": "not_verified", "reason": state.verification_reason}]
        state.verified = True
        state.verified_time = time
        state.verification_reason = "dwell satisfied and stair index is contiguous"
        state.transitions.append({"time": time, "state": "verified", "reason": state.verification_reason})
        self._next_verified_step += 1
        return [{"step": state.step, "time": time, "state": "verified", "reason": state.verification_reason}]

    def _event_dict(self, state: _StepState) -> dict[str, Any]:
        self._update_support_durations(state)
        endpoint_supports = [state.endpoint_supports[label] for label in sorted(state.endpoint_supports, key=lambda item: state.endpoint_supports[item]["time"])]
        endpoint_labels = [str(support["label"]) for support in endpoint_supports]
        endpoint_label = None
        if endpoint_supports:
            endpoint_label = endpoint_labels[0]
            if len(endpoint_supports) > 1 and float(endpoint_supports[1]["time"]) - float(endpoint_supports[0]["time"]) <= self.simultaneous_window_s:
                endpoint_label = "simultaneous"
        support_intervals = {}
        for label, run in state.runs.items():
            intervals = list(run.intervals)
            if run.start_time is not None:
                end_time = self._last_time if self._last_time is not None else run.start_time + run.duration_s
                intervals.append({"start_time": run.start_time, "end_time": end_time, "duration_s": run.duration_s, "peak_force_n": run.peak_force_n})
            support_intervals[label] = intervals
        return {
            "step": state.step,
            "first_touch_time": state.first_touch_time,
            "first_touch_label": state.first_touch_label,
            "first_touch_force_n": state.first_touch_force_n,
            "first_sustained_time": state.first_sustained_time,
            "first_sustained_label": state.first_sustained_label,
            "middle_support_time": state.middle_support_time,
            "middle_support_peak_force_n": state.middle_support_peak_force_n,
            "middle_preceded": state.middle_preceded,
            "endpoint_label": endpoint_label,
            "endpoint_supports": endpoint_supports,
            "ambiguous": state.ambiguous,
            "ambiguity_reasons": list(state.ambiguity_reasons),
            "verified": state.verified,
            "verified_time": state.verified_time,
            "verification_reason": state.verification_reason,
            "verification_attempted": state.verification_attempted,
            "support_intervals": support_intervals,
            "transitions": list(state.transitions),
        }

    @staticmethod
    def _flip_prefix(events: list[dict[str, Any]], *, require_unambiguous: bool) -> tuple[int, list[str]]:
        expected_step = 1
        previous_label: str | None = None
        sequence: list[str] = []
        for event in events:
            if not event["verified"]:
                continue
            if int(event["step"]) != expected_step:
                break
            label = event.get("endpoint_label")
            if label not in ENDPOINT_LABELS:
                break
            if previous_label is not None and label == previous_label:
                break
            if require_unambiguous and event.get("ambiguous", False):
                break
            sequence.append(str(label))
            previous_label = str(label)
            expected_step += 1
        return len(sequence), sequence


# A short alias keeps integration code readable while preserving the explicit
# class name for callers that want to make the sustained-support semantics clear.
SustainedSupportRecorder = GaitRecorder


__all__ = ["GaitRecorder", "SustainedSupportRecorder"]
