"""Reviewable demonstration and reference presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schemas import RunConfig


def _preset_config(scenario: str, **overrides: Any) -> RunConfig:
    material = overrides.pop("material", {})
    scene = overrides.pop("scene", {})
    numerics = overrides.pop("numerics", {})
    provenance = overrides.pop("provenance", {})
    return RunConfig(
        scenario=scenario,
        material=material,
        scene=scene,
        numerics=numerics,
        provenance=provenance or {"preset": "illustrative: not experimentally calibrated"},
        **overrides,
    )


def _definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "drop-validation",
            "label": "下落验证小算例（3 圈）",
            "description": "原生平衡、质心自由落体和16→32段收敛已检查；接触穿透与计算成本见验证报告。",
            "config": _preset_config(
                "drop", name="下落数值验证小算例",
                material={"turns": 3, "mass": 0.012, "pitch": 0.002, "young_modulus": 1e8, "shear_modulus": 1e8 / 2.7},
                numerics={"segments_per_turn": 16, "timestep": 0.0000125, "duration": 0.18, "sample_hz": 100},
                provenance={"preset": "drop-validation", "geometry_material": "explicit demonstration assumptions, not measured plastic slinky",
                            "numerical_evidence": "docs/validation-results.json; 16 to 32 segments and halved dt",
                            "experimental_support": "not established"},
            ).model_dump(mode="json"),
        },
        {
            "id": "drop-preview",
            "label": "自由下落（预览）",
            "description": "顶端约束静置后释放，用于检查质心自由落体与接触前动力学。",
            "config": _preset_config(
                "drop",
                name="彩虹圈自由下落预览",
                scene={"tilt_deg": 0.0},
                provenance={"preset": "drop-preview", "calibration": "illustrative: not experimentally calibrated"},
            ).model_dump(mode="json"),
        },
        {
            "id": "stairs-preview",
            "label": "翻转下楼梯（预览）",
            "description": "矩形截面螺旋在真实楼梯碰撞上推进，支持初始倾角与前向速度。",
            "config": _preset_config(
                "stairs",
                name="翻转下楼梯预览",
                scene={"tilt_deg": 25.0, "initial_forward_velocity": 0.08, "launch_offset": 0.0},
                provenance={"preset": "stairs-preview", "calibration": "illustrative: not experimentally calibrated"},
            ).model_dump(mode="json"),
        },
        {
            "id": "mujoco-coil-reference",
            "label": "MuJoCo coil 参考",
            "description": "采用官方 elasticity.cable 螺旋建模约定的可复现实验预设；参数仍需实物标定。",
            "config": _preset_config(
                "drop",
                name="MuJoCo coil 参考",
                scene={"tilt_deg": 0.0},
                material={"turns": 12, "radius": 0.03, "pitch": 0.0017, "young_modulus": 1.0e8, "shear_modulus": 3.7e7},
                numerics={"profile": "fine", "duration": 2.0, "segments_per_turn": 24, "timestep": 0.0001},
                provenance={
                    "preset": "mujoco-coil-reference",
                    "source": "MuJoCo 3.15.0 modeling.rst elasticity.cable example",
                    "calibration": "illustrative: not experimentally calibrated",
                },
            ).model_dump(mode="json"),
        },
        {
            "id": "literature-39-turn",
            "label": "文献 39 圈（静态拟合）",
            "description": "8段/圈静态悬挂长度拟合到1.137m；离散加密与下落时序需要独立验证，计算较慢。",
            "config": _preset_config(
                "drop",
                name="文献 39 圈独立预设",
                material={"turns": 39, "pitch": 0.066 / 39.0, "mass": 0.0487,
                          "young_modulus": 1422099858.232027, "shear_modulus": 526703651.197047},
                scene={"tilt_deg": 0.0},
                numerics={"profile": "preview", "duration": 0.35, "segments_per_turn": 8,
                          "timestep": 0.00001, "max_wall_seconds": 7200},
                provenance={
                    "preset": "literature-39-turn",
                    "source": "Cross and Wheatland (2012), arXiv:1208.4629",
                    "source_url": "https://arxiv.org/abs/1208.4629",
                    "source_doi": "10.1119/1.4750489",
                    "reported_observations": "plastic B: 39 turns, mass 0.0487 kg, compression 0.066 m, suspended length 1.14 m, fitted bulk stiffness 0.22 N/m",
                    "parameter_mapping": "pitch=0.066/39 is an explicit geometric approximation; radius, section, friction, Young modulus and shear modulus are not measured in the source",
                    "calibration": "close-coiled k to G mapping followed by native static-length fitting at 8 segments/turn; E=2.7G assumed",
                    "static_fit": "1.137366 m vs fitted target 1.14 m; relative error 0.231%; not independent experimental validation",
                },
            ).model_dump(mode="json"),
        },
    ]


def get_presets() -> list[dict[str, Any]]:
    """Return complete JSON-compatible preset records."""

    return deepcopy(_definitions())


def get_preset(preset_id: str) -> RunConfig:
    """Load one preset as a validated independent RunConfig."""

    for preset in _definitions():
        if preset["id"] == preset_id:
            return RunConfig.model_validate(deepcopy(preset["config"]))
    raise KeyError(f"unknown preset: {preset_id}")
