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
                material={"mass": 0.0487 * 12 / 39, "young_modulus": 4e8, "shear_modulus": 4e8 / 2.7},
                scene={"tilt_deg": 0.0},
                numerics={"segments_per_turn": 12, "timestep": 0.000025, "duration": 0.4},
                provenance={"preset": "drop-preview", "calibration": "illustrative: not experimentally calibrated",
                            "mass_assumption": "mass scaled from 39-turn reference by 12/39; geometry and E/G are assumptions",
                            "numerical_check": "12 segments, 25us timestep: 0.729% COM error at 0.4s; collapse not observed"},
            ).model_dump(mode="json"),
        },
        {
            "id": "stairs-preview",
            "label": "楼梯接触：停止（快速算例）",
            "description": "无初始扰动，0.8秒内停在起始踏面；实际停止检查通过，未证明翻转。",
            "config": _preset_config(
                "stairs",
                name="楼梯接触停止算例",
                material={"turns": 3, "strip_width": 0.006, "strip_thickness": 0.0045, "pitch": 0.0055,
                          "mass": 0.012, "young_modulus": 2e7, "shear_modulus": 2e7 / 2.7,
                          "damping": 0.002, "friction": 0.8, "self_friction": 0.05},
                scene={"step_depth": 0.12, "tilt_deg": 0.0, "initial_forward_velocity": 0.0,
                       "initial_angular_velocity": 0.0, "launch_offset": -0.015, "settle_time": 0.05},
                numerics={"profile": "fine", "duration": 0.8, "sample_hz": 50, "segments_per_turn": 8,
                          "timestep": 0.00005, "max_wall_seconds": 90, "contact_time_constant": 0.0005},
                provenance={"preset": "stairs-preview", "calibration": "explicit demonstration assumptions, no experimental support",
                            "numerical_observation": "stopped; quiet duration 0.66025s; penetration 0.0648mm; docs/validation/stairs-stopped.json"},
            ).model_dump(mode="json"),
        },
        {
            "id": "stairs-sliding",
            "label": "楼梯接触：滑移（12 圈）",
            "description": "触及五个下降踏面，端圈未连续交替；不能作为成功翻转预设，CPU计算较慢。",
            "config": _preset_config(
                "stairs", name="楼梯滑移对照算例",
                material={"strip_width": 0.006, "strip_thickness": 0.0045, "pitch": 0.0055,
                          "young_modulus": 2e7, "shear_modulus": 2e7 / 2.7, "friction": 0.8, "self_friction": 0.05},
                scene={"tilt_deg": 45.0, "initial_forward_velocity": 0.05, "initial_angular_velocity": 2.0,
                       "launch_offset": 0.0, "settle_time": 0.05},
                numerics={"profile": "fine", "duration": 2.4, "sample_hz": 20, "segments_per_turn": 8,
                          "timestep": 0.00005, "max_wall_seconds": 600, "contact_time_constant": 0.0005},
                provenance={"preset": "stairs-sliding", "calibration": "explicit demonstration assumptions, no experimental support",
                            "numerical_observation": "sliding; zero confirmed flips; docs/validation/stairs-world-axis-A.json"},
            ).model_dump(mode="json"),
        },
        {
            "id": "stairs-side-fall",
            "label": "楼梯接触：侧落（12 圈）",
            "description": "较高、较深台阶下发生侧落；未通过离散加密验收，CPU计算较慢。",
            "config": _preset_config(
                "stairs", name="楼梯侧落对照算例",
                material={"strip_width": 0.006, "strip_thickness": 0.0045, "pitch": 0.0055,
                          "young_modulus": 2e7, "shear_modulus": 2e7 / 2.7, "friction": 0.8, "self_friction": 0.05},
                scene={"step_height": 0.08, "step_depth": 0.12, "tilt_deg": 45.0,
                       "initial_forward_velocity": 0.05, "initial_angular_velocity": 2.0,
                       "launch_offset": 0.015, "settle_time": 0.05},
                numerics={"profile": "fine", "duration": 2.4, "sample_hz": 20, "segments_per_turn": 8,
                          "timestep": 0.00005, "max_wall_seconds": 600, "contact_time_constant": 0.0005},
                provenance={"preset": "stairs-side-fall", "calibration": "explicit demonstration assumptions, no experimental support",
                            "numerical_observation": "side_fall; zero confirmed flips; docs/validation/stairs-world-axis-B.json"},
            ).model_dump(mode="json"),
        },
        {
            "id": "stairs-walking",
            "label": "楼梯翻转：三阶基准（待加密确认）",
            "description": "65°摆放、0.12m阶深下末端→首端→末端支撑；粗离散已观察三阶，加密验收仍待完成。",
            "config": _preset_config(
                "stairs", name="楼梯三阶翻转基准",
                material={"strip_width": 0.006, "strip_thickness": 0.0045, "pitch": 0.0055,
                          "young_modulus": 2e7, "shear_modulus": 2e7 / 2.7, "friction": 0.8, "self_friction": 0.05},
                scene={"step_depth": 0.12, "tilt_deg": 65.0, "initial_forward_velocity": 0.05,
                       "initial_angular_velocity": 0.0, "launch_offset": 0.0, "settle_time": 0.05},
                numerics={"profile": "fine", "duration": 2.4, "sample_hz": 20, "segments_per_turn": 8,
                          "timestep": 0.00005, "max_wall_seconds": 600, "contact_time_constant": 0.0005},
                provenance={"preset": "stairs-walking", "calibration": "explicit demonstration assumptions, no experimental support",
                            "numerical_observation": "baseline three verified descending steps; mesh refinement pending; docs/validation/stairs-walking-baseline.json"},
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
