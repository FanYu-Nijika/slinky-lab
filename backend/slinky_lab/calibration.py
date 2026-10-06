"""Explicit approximate mapping from measured coil stiffness to rod moduli."""

import math

from .energy import rectangular_section_coefficients
from .schemas import Material


def moduli_from_coil_stiffness(material: Material, stiffness_n_m: float, poisson_ratio: float = 0.35) -> dict:
    """Use the close-coiled, small-load torsion approximation, then require fitting.

    The section and Poisson ratio are independent assumptions. A bulk N/m
    stiffness never becomes a Pa input directly. Large-pitch and discrete-rod
    corrections must be checked with a held native equilibrium calculation.
    """
    if not math.isfinite(stiffness_n_m) or stiffness_n_m <= 0:
        raise ValueError("coil stiffness must be finite and positive")
    if not math.isfinite(poisson_ratio) or not -1 < poisson_ratio < 0.5:
        raise ValueError("Poisson ratio must lie between -1 and 0.5")
    section = rectangular_section_coefficients((1, material.strip_width / 2, material.strip_thickness / 2))
    shear = stiffness_n_m * 2 * math.pi * material.radius ** 3 * material.turns / section["J"]
    young = 2 * (1 + poisson_ratio) * shear
    return {"bulk_stiffness_n_m": stiffness_n_m, "poisson_ratio_assumed": poisson_ratio,
            "shear_modulus_pa": shear, "young_modulus_pa": young, "torsion_constant_m4": section["J"],
            "method": "k = G*J / (2*pi*R^3*N), close-coiled torsion approximation",
            "status": "initial_estimate_requires_native_static_calibration"}
