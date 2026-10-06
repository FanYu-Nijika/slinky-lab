import math

import pytest

from slinky_lab.calibration import moduli_from_coil_stiffness
from slinky_lab.schemas import Material


def test_coil_mapping_reproduces_target_with_stated_section_and_poisson_assumption():
    material = Material(turns=39)
    result = moduli_from_coil_stiffness(material, 0.22)
    recovered = result["shear_modulus_pa"] * result["torsion_constant_m4"] / (2 * math.pi * material.radius ** 3 * 39)
    assert recovered == pytest.approx(0.22)
    assert result["young_modulus_pa"] == pytest.approx(2.7 * result["shear_modulus_pa"])
    assert result["shear_modulus_pa"] > 1e8
    assert result["status"] == "initial_estimate_requires_native_static_calibration"


def test_invalid_bulk_stiffness_is_rejected():
    with pytest.raises(ValueError):
        moduli_from_coil_stiffness(Material(), 0)
