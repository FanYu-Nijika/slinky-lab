import math

import pytest
from pydantic import ValidationError

from slinky_lab.schemas import Frame, RunConfig, SweepConfig


def test_physical_configuration_rejects_nonfinite_and_interpenetrating_pitch():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"material": {"mass": math.nan}})
    with pytest.raises(ValidationError, match="pitch"):
        RunConfig.model_validate({"material": {"pitch": 0.001, "strip_thickness": 0.002}})


def test_geometry_and_sweep_resource_limits():
    with pytest.raises(ValidationError, match="2048"):
        RunConfig.model_validate({"material": {"turns": 80}, "numerics": {"segments_per_turn": 64}})
    with pytest.raises(ValidationError, match="different"):
        SweepConfig.model_validate({"base": {}, "axes": [
            {"parameter": "material.mass", "values": [0.1]},
            {"parameter": "material.mass", "values": [0.2]},
        ]})


def test_frames_reject_nonfinite_and_inconsistent_geometry():
    with pytest.raises(ValidationError):
        Frame(time=0, positions=[[0, 0, math.inf]], quaternions=[[1, 0, 0, 0]])
    with pytest.raises(ValidationError, match="orientation"):
        Frame(time=0, positions=[[0, 0, 0]], quaternions=[])
