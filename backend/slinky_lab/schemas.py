"""Shared, SI-unit contracts for the solver, API, CLI, and browser."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Material(StrictModel):
    turns: int = Field(12, ge=3, le=80)
    radius: float = Field(0.03, ge=0.005, le=0.15)
    strip_width: float = Field(0.003, ge=0.0002, le=0.02)
    strip_thickness: float = Field(0.0015, ge=0.0002, le=0.008)
    pitch: float = Field(0.0017, ge=0.0002, le=0.02)
    mass: float = Field(0.0487, ge=0.001, le=2)
    young_modulus: float = Field(1.0e8, ge=1.0e4, le=3.0e11)
    shear_modulus: float = Field(3.7e7, ge=1.0e3, le=1.5e11)
    damping: float = Field(0.00001, ge=0, le=0.1)
    friction: float = Field(0.5, ge=0, le=3)
    self_friction: float = Field(0.5, ge=0, le=3)

    @model_validator(mode="after")
    def valid_geometry(self):
        if self.pitch < self.strip_thickness:
            raise ValueError("pitch must be at least strip_thickness")
        if self.strip_width >= self.radius:
            raise ValueError("strip_width must be smaller than radius")
        return self


class Scene(StrictModel):
    gravity: float = Field(9.81, ge=0, le=30)
    step_height: float = Field(0.04, ge=0.002, le=0.3)
    step_depth: float = Field(0.08, ge=0.015, le=0.5)
    step_width: float = Field(0.3, ge=0.05, le=2)
    step_count: int = Field(6, ge=3, le=20)
    tilt_deg: float = Field(25, ge=-80, le=80)
    initial_pose: Literal["tilted", "arched"] = "tilted"
    arch_rise: float | None = Field(None, ge=0, le=1)
    arch_end_turns: int = Field(2, ge=1, le=20)
    arch_free_clearance: float = Field(0.025, ge=0, le=0.5)
    initial_angular_velocity: float = Field(0, ge=-20, le=20)
    initial_forward_velocity: float = Field(0, ge=-2, le=2)
    initial_lateral_velocity: float = Field(0, ge=-2, le=2)
    launch_offset: float = Field(0, ge=-0.1, le=0.2)
    settle_time: float = Field(2, ge=0.05, le=15)


class Numerics(StrictModel):
    profile: Literal["preview", "fine"] = "preview"
    duration: float = Field(1.5, ge=0.02, le=30)
    sample_hz: int = Field(60, ge=10, le=120)
    segments_per_turn: int | None = Field(None, ge=8, le=64)
    timestep: float | None = Field(None, ge=0.000001, le=0.002)
    max_wall_seconds: float = Field(600, ge=5, le=14400)
    contact_time_constant: float = Field(0.003, ge=0.00005, le=0.03)

    def resolved(self):
        return {
            "segments_per_turn": self.segments_per_turn or (12 if self.profile == "preview" else 24),
            "timestep": self.timestep or (0.0002 if self.profile == "preview" else 0.00005),
            "iterations": 50 if self.profile == "preview" else 100,
            "tolerance": 1e-8 if self.profile == "preview" else 1e-10,
        }


class RunConfig(StrictModel):
    schema_version: Literal[1] = 1
    name: str = Field("彩虹圈实验", max_length=100)
    scenario: Literal["drop", "stairs"] = "drop"
    material: Material = Field(default_factory=Material)
    scene: Scene = Field(default_factory=Scene)
    numerics: Numerics = Field(default_factory=Numerics)
    provenance: dict[str, str] = Field(default_factory=lambda: {"preset": "illustrative: not experimentally calibrated"})

    @model_validator(mode="after")
    def bounded_mesh(self):
        if self.material.turns * self.numerics.resolved()["segments_per_turn"] > 2048:
            raise ValueError("The CPU workbench supports at most 2048 rod segments per run")
        if self.scene.initial_pose == "arched":
            if self.scenario != "stairs":
                raise ValueError("initial_pose='arched' is only valid for the stairs scenario")
            if self.material.turns < 2 * self.scene.arch_end_turns + 2:
                raise ValueError("arched initial pose requires turns >= 2*arch_end_turns + 2")
        return self


class Contact(StrictModel):
    position: list[float] = Field(min_length=3, max_length=3)
    geom_a: int = Field(ge=0)
    geom_b: int = Field(ge=0)
    normal_force: float = Field(ge=0)
    kind: Literal["self", "stair", "external"]
    stair_step: int | None = Field(None, ge=0)
    material_index: int | None = Field(None, ge=0)
    surface: Literal["tread", "other"] = "other"


class Frame(StrictModel):
    time: float
    positions: list[list[float]]
    quaternions: list[list[float]]
    contacts: list[list[float]] = Field(default_factory=list)
    contact_details: list[Contact] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_geometry(self):
        if len(self.positions) != len(self.quaternions):
            raise ValueError("Every rod position must have an orientation")
        if any(len(p) != 3 for p in self.positions) or any(len(q) != 4 for q in self.quaternions):
            raise ValueError("Positions must be xyz and quaternions must be wxyz")
        if any(len(p) != 3 for p in self.contacts):
            raise ValueError("Contact positions must be xyz")
        return self


class RunResult(StrictModel):
    run_id: str
    status: str
    config: RunConfig
    summary: dict = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class Command(StrictModel):
    action: Literal["pause", "resume", "step", "cancel"]


class SweepAxis(StrictModel):
    parameter: str
    values: list[float] = Field(min_length=1, max_length=20)


class SweepConfig(StrictModel):
    name: str = Field("参数扫描", max_length=100)
    base: RunConfig
    axes: list[SweepAxis] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def bounded_grid(self):
        size = 1
        for axis in self.axes:
            size *= len(axis.values)
        if size > 100:
            raise ValueError("A sweep is limited to 100 runs")
        if len({axis.parameter for axis in self.axes}) != len(self.axes):
            raise ValueError("Sweep axes must be different parameters")
        return self
