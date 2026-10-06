"""MuJoCo 3.15.0 cable model and small-step simulation adapter.

The model deliberately uses MuJoCo's first-party elasticity.cable plugin.  The
plugin receives a curved, box-section composite cable so that the rest shape,
anisotropic rectangular section, contacts, and inextensibility are all handled
by the same engine as the final application.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any, Callable

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - the package is a project dependency
    mujoco = None  # type: ignore[assignment]

from .schemas import Frame, RunConfig


ProgressCallback = Callable[[float], None]
CancelCallback = Callable[[], bool]


def _xml_float(value: float) -> str:
    """Format a finite scalar compactly for MJCF."""

    if not math.isfinite(float(value)):
        raise ValueError("model parameters must be finite")
    return f"{float(value):.12g}"


def _vec3(values: tuple[float, float, float] | list[float] | np.ndarray) -> str:
    return " ".join(_xml_float(float(value)) for value in values)


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=float,
    )


def _rotation_y_quat(angle_rad: float) -> tuple[float, float, float, float]:
    half = angle_rad * 0.5
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to MuJoCo's wxyz quaternion."""

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale],
            dtype=float,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale],
                dtype=float,
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                [(matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale],
                dtype=float,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])) * 2.0
            quaternion = np.array(
                [(matrix[1, 0] - matrix[0, 1]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale],
                dtype=float,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0 or not math.isfinite(norm):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quaternion / norm


class Simulation:
    """One prepared MuJoCo experiment with a fixed SI-unit configuration."""

    def __init__(self, config: RunConfig):
        if mujoco is None:
            raise RuntimeError("MuJoCo 3.15.0 is required to create a Simulation")
        self.config = config
        self._resolved = config.numerics.resolved()
        self.dt = float(self._resolved["timestep"])
        self.duration = float(config.numerics.duration)
        self._segments = config.material.turns * int(self._resolved["segments_per_turn"])
        self._height = config.material.pitch * config.material.turns
        # A drop starts vertically.  The shared Scene schema also serves the
        # stairs run, whose tilt is meaningful there but would silently bias a
        # free-fall check if it were reused for the drop preset.
        self._angle = math.radians(config.scene.tilt_deg) if config.scenario == "stairs" else 0.0
        self._offset = self._initial_offset()
        self._anchor = self._cable_endpoint()
        self.model_xml = self._build_model_xml()
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        self._body_ids, self._geom_ids = self._discover_cable_elements()
        self._cable_geom_to_material = {geom_id: index for index, geom_id in enumerate(self._geom_ids)}
        self._stair_geom_ids = {
            geom_id for geom_id in range(int(self.model.ngeom))
            if self.model.geom(geom_id).name.startswith("stair_")
        }
        self._stair_geom_to_step = {
            geom_id: int(self.model.geom(geom_id).name.split("_")[-1])
            for geom_id in self._stair_geom_ids
        }
        self._half_sizes = np.asarray(self.model.geom_size[self._geom_ids], dtype=float).copy()
        self._model_mass = float(np.sum(np.asarray(self.model.body_mass[self._body_ids], dtype=float)))
        self._previous_quaternions: np.ndarray | None = None
        self._prepared = False
        self._released = config.scenario != "drop"
        self._status = "created"
        self._steps = 0
        self._cable_work = 0.0
        self._damping_work = 0.0
        self._contact_work = 0.0
        self._max_penetration = 0.0
        self._nonadjacent_self_contacts = 0
        self._stair_contact_events = 0
        self._max_nonadjacent_self_contacts = 0
        self._max_stair_contacts = 0
        self._last_stair_steps: set[int] = set()
        self._max_step = 0
        self._last_stair_contact_labels: dict[int, set[str]] = {}
        self._step_events: list[dict[str, Any]] = []
        self._settle_speed = 0.0
        self._settle_damping = 0.0
        self._settle_steps = 0
        self._settle_residual_acceleration = float("nan")
        self._settle_status = "not_run"
        self._release_com_z: float | None = None
        self._release_com_vz: float | None = None
        self._release_bottom_z: float | None = None
        self._initial_hanging_length: float | None = None
        self._static_equilibrium_length: float | None = None
        self._bottom_onset_time: float | None = None
        self._collapse_time: float | None = None
        self._bottom_onset_threshold: float | None = None
        self._collapse_threshold: float | None = None
        self._collapse_initially_elongated = False
        self._top_endpoint_index = 0
        self._bottom_endpoint_index = -1
        self._freefall_error = float("nan")
        self._freefall_samples = 0
        self._initialise_state()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def _initial_offset(self) -> tuple[float, float, float]:
        material = self.config.material
        scene = self.config.scene
        if self.config.scenario == "drop":
            z = max(0.28, scene.step_height * scene.step_count + 0.12)
            return (0.0, 0.0, z)
        top_height = scene.step_height * scene.step_count
        min_z, _ = self._rest_world_extents()[2]
        clearance = max(0.001, material.strip_thickness)
        return (scene.step_depth * 0.5 + scene.launch_offset, 0.0, top_height - min_z + clearance)

    def _rest_world_extents(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Return world-frame rest-shape bounds relative to the composite offset."""

        material = self.config.material
        samples = np.linspace(0.0, 1.0, max(256, self._segments + 1))
        phase = 2.0 * math.pi * material.turns * samples
        local = np.column_stack((material.radius * np.cos(phase), material.radius * np.sin(phase), self._height * samples))
        world = local @ _rotation_y(self._angle).T
        bounds = []
        for axis in range(3):
            bounds.append((float(np.min(world[:, axis])), float(np.max(world[:, axis]))))
        return tuple(bounds)  # type: ignore[return-value]

    def _cable_endpoint(self) -> tuple[float, float, float]:
        rotation = _rotation_y(self._angle)
        endpoint = rotation @ np.array([self.config.material.radius, 0.0, self._height], dtype=float)
        return tuple(np.asarray(self._offset, dtype=float) + endpoint)

    def _build_composite_model_xml(self) -> str:
        material = self.config.material
        scene = self.config.scene
        numerics = self._resolved
        angle_quat = _rotation_y_quat(self._angle)
        segment_length = math.sqrt(
            (2.0 * math.pi * material.turns * material.radius) ** 2 + self._height**2
        ) / self._segments
        half_length = max(segment_length * 0.52, material.strip_width * 0.75)
        half_width = material.strip_width * 0.5
        half_thickness = material.strip_thickness * 0.5
        segment_mass = material.mass / self._segments
        gravity = _vec3((0.0, 0.0, -float(scene.gravity)))
        friction = _vec3((material.friction, material.friction * 0.8, material.friction * 0.02))
        quat = _vec3(angle_quat)
        offset = _vec3(self._offset)
        # The cable compiler evaluates cos/sin at pi * size[2] * s.  A
        # complete turn therefore needs size[2] = 2 * turns.
        size = _vec3((self._height, material.radius, float(2 * material.turns)))
        box_size = _vec3((half_length, half_width, half_thickness))
        cable_plugin = """<plugin plugin=\"mujoco.elasticity.cable\">
          <config key=\"twist\" value=\"{twist}\"/>
          <config key=\"bend\" value=\"{bend}\"/>
          <config key=\"flat\" value=\"false\"/>
        </plugin>""".format(
            twist=_xml_float(material.shear_modulus), bend=_xml_float(material.young_modulus)
        )
        stairs = self._stairs_xml() if self.config.scenario == "stairs" else ""
        hold = ""
        anchor_body = ""
        if self.config.scenario == "drop":
            anchor_body = f'<body name="drop_anchor" pos="{_vec3(self._anchor)}"/>'
            hold = """<equality>
          <connect name=\"drop_hold\" body1=\"B_last\" body2=\"drop_anchor\" anchor=\"0 0 0\"/>
        </equality>"""
        ground_z = -100.0 if self.config.scenario == "drop" else 0.0
        return f"""<mujoco model=\"slinky_lab\">
      <compiler angle=\"radian\" coordinate=\"local\" autolimits=\"true\"/>
      <option timestep=\"{_xml_float(self.dt)}\" gravity=\"{gravity}\" integrator=\"implicitfast\"
              iterations=\"{int(numerics['iterations'])}\" tolerance=\"{_xml_float(numerics['tolerance'])}\">
        <flag energy=\"enable\"/>
      </option>
      <size nconmax=\"{max(2000, self._segments * 12)}\" njmax=\"{max(2000, self._segments * 8)}\"/>
      <default>
        <joint damping=\"{_xml_float(material.damping)}\" armature=\"0\"/>
        <geom contype=\"1\" conaffinity=\"1\" friction=\"{friction}\"
              solref=\"0.003 1\" solimp=\"0.95 0.99 0.001 0.5 2\"/>
      </default>
      <extension><plugin plugin=\"mujoco.elasticity.cable\"/></extension>
      <worldbody>
        <geom name=\"ground\" type=\"plane\" pos=\"0 0 {_xml_float(ground_z)}\" size=\"100 100 0.01\"
              contype=\"1\" conaffinity=\"1\"/>
        {stairs}
        {anchor_body}
        <composite type=\"cable\" curve=\"cos(s) sin(s) s\"
                   count=\"{self._segments + 1} 1 1\" size=\"{size}\" offset=\"{offset}\"
                   quat=\"{quat}\" initial=\"free\">
          {cable_plugin}
          <joint kind=\"main\" damping=\"{_xml_float(material.damping)}\" armature=\"0\"/>
          <geom type=\"box\" size=\"{box_size}\" mass=\"{_xml_float(segment_mass)}\"
                friction=\"{friction}\" rgba=\"0.1 0.6 0.8 1\" group=\"3\"/>
        </composite>
      </worldbody>
      {hold}
    </mujoco>"""

    def _build_model_xml(self) -> str:
        """Build an explicit radial-frame chain around the cable plugin.

        The stock composite generator uses a Bishop frame, which is valid for
        isotropic rods but can rotate a rectangular strip around a tight helix.
        Explicit body frames keep the strip width radial and the thickness
        normal to the helix while retaining the first-party cable plugin on
        every element.
        """

        root = ET.fromstring(self._build_composite_model_xml())
        world = root.find("worldbody")
        composite = world.find("composite") if world is not None else None
        extension_plugin = root.find("extension/plugin")
        if world is None or composite is None or extension_plugin is None:
            raise RuntimeError("failed to locate cable model template")
        world.remove(composite)
        instance = ET.SubElement(extension_plugin, "instance", {"name": "rod_material"})
        source_plugin = composite.find("plugin")
        if source_plugin is None:
            raise RuntimeError("cable plugin configuration is missing")
        for config in source_plugin.findall("config"):
            instance.append(ET.fromstring(ET.tostring(config, encoding="unicode")))

        material = self.config.material
        theta = np.linspace(0.0, 2.0 * math.pi * material.turns, self._segments + 1)
        local_vertices = np.column_stack((
            material.radius * np.cos(theta),
            material.radius * np.sin(theta),
            self._height * np.linspace(0.0, 1.0, self._segments + 1),
        ))
        rotation = _rotation_y(self._angle)
        vertices = local_vertices @ rotation.T + np.asarray(self._offset, dtype=float)
        parent = world
        previous_frame: np.ndarray | None = None
        previous_name: str | None = None
        contact = ET.SubElement(root, "contact")

        for index in range(self._segments):
            tangent = vertices[index + 1] - vertices[index]
            length = float(np.linalg.norm(tangent))
            if length <= 0.0:
                raise ValueError("cable discretization produced a zero-length element")
            tangent /= length
            phase = 0.5 * (theta[index] + theta[index + 1])
            width = rotation @ np.array([-math.cos(phase), -math.sin(phase), 0.0], dtype=float)
            width -= tangent * float(np.dot(width, tangent))
            width /= np.linalg.norm(width)
            normal = np.cross(tangent, width)
            normal /= np.linalg.norm(normal)
            width = np.cross(normal, tangent)
            frame = np.column_stack((tangent, width, normal))
            body_name = "B_first" if index == 0 else ("B_last" if index == self._segments - 1 else f"B_{index}")
            if previous_frame is None:
                body_pos = vertices[index]
                body_frame = frame
            else:
                body_pos = previous_frame.T @ (vertices[index] - vertices[index - 1])
                body_frame = previous_frame.T @ frame
            body = ET.SubElement(parent, "body", {
                "name": body_name,
                "pos": _vec3(body_pos),
                "quat": _vec3(_matrix_to_quat(body_frame)),
            })
            if index == 0:
                ET.SubElement(body, "freejoint", {"name": "J_first"})
            else:
                ET.SubElement(body, "joint", {
                    "name": f"J_{index}",
                    "type": "ball",
                    "damping": _xml_float(material.damping),
                    "armature": "0",
                })
            ET.SubElement(body, "geom", {
                "name": f"G{index}",
                "type": "box",
                "pos": _vec3((length * 0.5, 0.0, 0.0)),
                "size": _vec3((length * 0.5, material.strip_width * 0.5, material.strip_thickness * 0.5)),
                "mass": _xml_float(material.mass / self._segments),
            })
            ET.SubElement(body, "plugin", {"instance": "rod_material"})
            if index == 0:
                ET.SubElement(body, "site", {"name": "S_first", "pos": "0 0 0", "size": ".001"})
            if index == self._segments - 1:
                ET.SubElement(body, "site", {"name": "S_last", "pos": _vec3((length, 0.0, 0.0)), "size": ".001"})
            if previous_name is not None:
                ET.SubElement(contact, "exclude", {"body1": previous_name, "body2": body_name})
            previous_frame = frame
            previous_name = body_name
            parent = body
        return ET.tostring(root, encoding="unicode")

    def _stairs_xml(self) -> str:
        scene = self.config.scene
        boxes: list[str] = []
        for index in range(scene.step_count):
            height = scene.step_height * (scene.step_count - index)
            x = scene.step_depth * (index + 0.5)
            half_size = (scene.step_depth * 0.5, scene.step_width * 0.5, height * 0.5)
            boxes.append(
                f'<geom name="stair_{index}" type="box" pos="{_vec3((x, 0.0, height * 0.5))}" '
                f'size="{_vec3(half_size)}" rgba="0.35 0.35 0.38 1"/>'
            )
        return "\n        ".join(boxes)

    def _discover_cable_elements(self) -> tuple[list[int], list[int]]:
        body_plugin = getattr(self.model, "body_plugin", None)
        if body_plugin is None:
            candidates = list(range(1, int(self.model.nbody)))
        else:
            plugins = np.asarray(body_plugin, dtype=int)
            candidates = [index for index in range(1, int(self.model.nbody)) if plugins[index] >= 0]
        candidates = [
            index for index in candidates
            if int(self.model.body_geomnum[index]) > 0 and int(self.model.body_parentid[index]) >= 0
        ]
        candidates.sort()
        body_ids: list[int] = []
        geom_ids: list[int] = []
        for body_id in candidates:
            geom_start = int(self.model.body_geomadr[body_id])
            geom_count = int(self.model.body_geomnum[body_id])
            if geom_count:
                body_ids.append(body_id)
                geom_ids.append(geom_start)
        if len(geom_ids) != self._segments:
            names = [self.model.body(i).name for i in range(1, int(self.model.nbody))]
            generated = [i for i, name in enumerate(names, start=1) if name.startswith("B_")]
            generated_geom_ids = [
                int(self.model.body_geomadr[i]) for i in generated if int(self.model.body_geomnum[i])
            ]
            generated = [i for i in generated if int(self.model.body_geomnum[i])]
            if len(generated_geom_ids) == self._segments:
                body_ids, geom_ids = generated, generated_geom_ids
        if len(geom_ids) != self._segments:
            raise RuntimeError(f"expected {self._segments} cable elements, found {len(geom_ids)}")
        return body_ids, geom_ids

    def _initialise_state(self) -> None:
        if self.config.scenario == "drop" and int(self.model.neq):
            self.data.eq_active[0] = 1
        root_body = self._body_ids[0]
        joint_start = int(self.model.body_jntadr[root_body])
        joint_count = int(self.model.body_jntnum[root_body])
        if joint_count:
            joint_id = joint_start
            dof_start = int(self.model.jnt_dofadr[joint_id])
            dof_count = self._joint_dof_count(joint_id)
            if dof_count >= 6:
                self.data.qvel[dof_start] = self.config.scene.initial_forward_velocity
                self.data.qvel[dof_start + 4] = self.config.scene.initial_angular_velocity
        mujoco.mj_forward(self.model, self.data)

    def _joint_dof_count(self, joint_id: int) -> int:
        joint_type = int(self.model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            return 6
        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            return 3
        return 1

    @staticmethod
    def _is_cancelled(callback: CancelCallback | None) -> bool:
        return bool(callback and callback())

    @staticmethod
    def _report_progress(callback: ProgressCallback | Callable[[float], None] | None, phase: str, value: float) -> None:
        if callback is None:
            return
        progress = float(max(0.0, min(1.0, value)))
        try:
            callback(progress)
        except TypeError:
            callback({"phase": phase, "progress": progress})  # type: ignore[arg-type]

    def prepare(
        self,
        progress: ProgressCallback | Callable[[float], None] | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> None:
        """Prepare the initial pose and release a temporarily held drop cable."""

        if self._prepared:
            return
        if self.config.scenario == "drop":
            settle_steps = max(1, int(math.ceil(self.config.scene.settle_time / self.dt)))
            self._status = "settling"
            self.data.eq_active[0] = 1
            physical_damping = np.asarray(self.model.dof_damping, dtype=float).copy()
            self._settle_damping = 5.0
            settle_damping = np.maximum(physical_damping, self._settle_damping)
            self.model.dof_damping[:] = settle_damping
            cancelled = False
            unstable = False
            stable_steps = 0
            stable_required = min(100, max(10, settle_steps // 20))
            for index in range(settle_steps):
                if self._is_cancelled(should_cancel):
                    cancelled = True
                    break
                mujoco.mj_step(self.model, self.data)
                if not np.all(np.isfinite(np.asarray(self.data.qvel, dtype=float))):
                    unstable = True
                    break
                settle_speed = float(np.linalg.norm(np.asarray(self.data.qvel, dtype=float)))
                stable_steps = stable_steps + 1 if settle_speed <= 0.05 else 0
                self._settle_steps = index + 1
                if index % max(1, settle_steps // 100) == 0:
                    self._report_progress(progress, "settle", (index + 1) / settle_steps)
                if stable_steps >= stable_required:
                    break
            self.model.dof_damping[:] = physical_damping
            if cancelled:
                self._status = "cancelled"
                return
            if unstable:
                self._status = "not_converged"
                raise RuntimeError("drop settling became numerically unstable")
            self.data.eq_active[0] = 0
            self.data.time = 0.0
            mujoco.mj_forward(self.model, self.data)
            qacc = np.asarray(self.data.qacc, dtype=float)
            self._settle_residual_acceleration = float(np.max(np.abs(qacc))) if qacc.size else 0.0
            self._settle_speed = float(np.linalg.norm(np.asarray(self.data.qvel, dtype=float)))
            self._settle_status = "converged" if self._settle_speed <= 0.05 and self._settle_residual_acceleration <= 1.0 else "not_converged"
            self._release_com_z = self._com_position()[2]
            self._release_com_vz = self._com_velocity()[2]
            self._released = True
            self._prepare_research_metrics()
        else:
            mujoco.mj_forward(self.model, self.data)
            self._settle_status = "not_applicable"
            self._prepare_research_metrics()
            self._report_progress(progress, "prepare", 1.0)
        self._prepared = True
        self._status = "ready"

    def _joint_damping_power(self) -> float:
        qvel = np.asarray(self.data.qvel, dtype=float)
        damping = np.asarray(self.model.dof_damping, dtype=float)
        return float(np.dot(damping, qvel**2))

    def _update_work(self) -> None:
        qvel = np.asarray(self.data.qvel, dtype=float)
        passive = np.asarray(self.data.qfrc_passive, dtype=float)
        damping_power = self._joint_damping_power()
        passive_power = float(np.dot(passive, qvel))
        cable_power = passive_power + damping_power
        constraint = getattr(self.data, "qfrc_constraint", None)
        contact_power = float(np.dot(np.asarray(constraint, dtype=float), qvel)) if constraint is not None else 0.0
        self._cable_work += cable_power * self.dt
        self._damping_work -= damping_power * self.dt
        self._contact_work += contact_power * self.dt

    def _update_contact_diagnostics(self) -> None:
        nonadjacent = 0
        stair_contacts = 0
        self._last_stair_steps = set()
        self._last_stair_contact_labels = {}
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            self._max_penetration = max(self._max_penetration, max(0.0, -float(contact.dist)))
            geom_a, geom_b = int(contact.geom[0]), int(contact.geom[1])
            material_a = self._cable_geom_to_material.get(geom_a)
            material_b = self._cable_geom_to_material.get(geom_b)
            if material_a is not None and material_b is not None and abs(material_a - material_b) > 1:
                nonadjacent += 1
            elif (geom_a in self._stair_geom_ids) != (geom_b in self._stair_geom_ids):
                stair_contacts += 1
                stair_geom = geom_a if geom_a in self._stair_geom_ids else geom_b
                step_index = self._stair_geom_to_step[stair_geom]
                self._last_stair_steps.add(step_index)
                cable_geom = geom_b if stair_geom == geom_a else geom_a
                material_index = self._cable_geom_to_material.get(cable_geom, -1)
                end_fraction = max(1, len(self._geom_ids) // 10)
                if material_index < end_fraction:
                    label = "first"
                elif material_index >= len(self._geom_ids) - end_fraction:
                    label = "last"
                else:
                    label = "middle"
                self._last_stair_contact_labels.setdefault(step_index, set()).add(label)
        self._nonadjacent_self_contacts += nonadjacent
        self._stair_contact_events += stair_contacts
        self._max_nonadjacent_self_contacts = max(self._max_nonadjacent_self_contacts, nonadjacent)
        self._max_stair_contacts = max(self._max_stair_contacts, stair_contacts)

    def _com_position(self) -> np.ndarray:
        masses = np.asarray(self.model.body_mass[self._body_ids], dtype=float)
        positions = np.asarray(self.data.xipos[self._body_ids], dtype=float)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            return np.zeros(3, dtype=float)
        return np.sum(positions * masses[:, None], axis=0) / total_mass

    def _com_velocity(self) -> np.ndarray:
        # cvel is expressed about each subtree COM, so mass-weighting body
        # linear slices would add rotational offsets.  The cable is one root
        # subtree; MuJoCo's subtree velocity is already the COM velocity.
        mujoco.mj_subtreeVel(self.model, self.data)
        root_body = self._body_ids[0]
        return np.asarray(self.data.subtree_linvel[root_body], dtype=float).copy()

    def _has_external_contact(self) -> bool:
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom_a, geom_b = int(contact.geom[0]), int(contact.geom[1])
            if not (geom_a in self._cable_geom_to_material and geom_b in self._cable_geom_to_material):
                return True
        return False

    def _material_vertical_span(self) -> float:
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        return float(abs(positions[-1, 2] - positions[0, 2]))

    def _prepare_research_metrics(self) -> None:
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        endpoints = positions[[0, -1], 2]
        self._top_endpoint_index = 0 if endpoints[0] >= endpoints[1] else -1
        self._bottom_endpoint_index = -1 if self._top_endpoint_index == 0 else 0
        self._static_equilibrium_length = self._material_vertical_span()
        self._initial_hanging_length = self._static_equilibrium_length
        self._bottom_onset_threshold = max(0.0005, 0.005 * self._initial_hanging_length)
        self._collapse_threshold = 1.05 * self.config.material.turns * self.config.material.strip_thickness
        self._collapse_initially_elongated = self._initial_hanging_length > self._collapse_threshold
        self._release_bottom_z = float(positions[self._bottom_endpoint_index, 2])

    def _update_research_metrics(self) -> None:
        if self.config.scenario != "drop" or self._release_bottom_z is None:
            return
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        bottom_z = float(positions[self._bottom_endpoint_index, 2])
        if self._bottom_onset_time is None and self._bottom_onset_threshold is not None:
            if abs(bottom_z - self._release_bottom_z) >= self._bottom_onset_threshold:
                self._bottom_onset_time = self.time
        if self._collapse_time is None and self._collapse_initially_elongated and self._collapse_threshold is not None:
            if self._material_vertical_span() <= self._collapse_threshold:
                self._collapse_time = self.time

    def _update_freefall(self) -> None:
        if self.config.scenario != "drop" or self._release_com_z is None or self._has_external_contact():
            return
        if self.time <= 0.2:
            initial_vz = self._release_com_vz or 0.0
            expected = self._release_com_z + initial_vz * self.time - 0.5 * self.config.scene.gravity * self.time**2
            error = abs(self._com_position()[2] - expected)
            self._freefall_error = max(error, self._freefall_error if math.isfinite(self._freefall_error) else 0.0)
            self._freefall_samples += 1

    def _update_steps(self) -> None:
        if self.config.scenario != "stairs":
            return
        # A step is counted only after a cable/stair contact is observed.  A
        # centre-of-mass crossing by itself would label a flight over a stair
        # as a descent even when no support transfer occurred.
        if self._last_stair_steps:
            for step_index in sorted(self._last_stair_steps):
                if step_index > self._max_step:
                    labels = self._last_stair_contact_labels.get(step_index, {"middle"})
                    if "first" in labels and "last" in labels:
                        end = "both"
                    elif "first" in labels:
                        end = "first"
                    elif "last" in labels:
                        end = "last"
                    else:
                        end = "middle"
                    self._step_events.append({"time": self.time, "step": step_index, "end": end})
                    self._max_step = step_index

    def _movement_classification(self) -> str:
        labels = [event["end"] for event in self._step_events if event["end"] in {"first", "last"}]
        if len(labels) >= 2 and all(left != right for left, right in zip(labels, labels[1:])):
            return "flip"
        if self._step_events or self._stair_contact_events:
            return "sliding"
        return "no_step_contact"

    def step(self) -> None:
        """Advance exactly one MuJoCo physical step."""

        if not self._prepared:
            self.prepare()
        if self._status == "cancelled" or self.time >= self.duration:
            self._status = "complete" if self.time >= self.duration else self._status
            return
        mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(np.asarray(self.data.qpos, dtype=float))) or not np.all(np.isfinite(np.asarray(self.data.qvel, dtype=float))):
            self._status = "unstable"
            raise RuntimeError("MuJoCo state became non-finite")
        self._steps += 1
        self._update_work()
        self._update_contact_diagnostics()
        # mj_step integrates qpos after evaluating the dynamics.  Recompute
        # kinematics so the frame pose and its timestamp describe one state;
        # the full forward pass is deferred to sampled frame/summary calls.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)
        self._update_freefall()
        self._update_research_metrics()
        self._update_steps()
        if self.time >= self.duration:
            self._status = "complete"
        else:
            self._status = "running"

    def _render_quaternions(self) -> np.ndarray:
        matrices = np.asarray(self.data.geom_xmat[self._geom_ids], dtype=float).reshape(-1, 3, 3)
        quaternions = np.vstack([_matrix_to_quat(matrix) for matrix in matrices])
        if self._previous_quaternions is not None:
            signs = np.sum(quaternions * self._previous_quaternions, axis=1) < 0.0
            quaternions[signs] *= -1.0
        self._previous_quaternions = quaternions.copy()
        return quaternions

    def _metrics(self) -> dict[str, float]:
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        matrices = np.asarray(self.data.geom_xmat[self._geom_ids], dtype=float).reshape(-1, 3, 3)
        extents = np.sum(np.abs(matrices[:, 2, :]) * self._half_sizes, axis=1)
        com = self._com_position()
        com_velocity = self._com_velocity()
        spatial_top_z = float(np.max(positions[:, 2] + extents))
        spatial_bottom_z = float(np.min(positions[:, 2] - extents))
        material_top_z = float(positions[self._top_endpoint_index, 2])
        material_bottom_z = float(positions[self._bottom_endpoint_index, 2])
        top_z = material_top_z
        bottom_z = material_bottom_z
        mujoco.mj_energyPos(self.model, self.data)
        mujoco.mj_energyVel(self.model, self.data)
        return {
            "top_z": top_z,
            "bottom_z": bottom_z,
            "spatial_top_z": spatial_top_z,
            "spatial_bottom_z": spatial_bottom_z,
            "material_top_z": material_top_z,
            "material_bottom_z": material_bottom_z,
            "material_first_z": float(positions[0, 2]),
            "material_last_z": float(positions[-1, 2]),
            "material_vertical_span": self._material_vertical_span(),
            "com_x": float(com[0]),
            "com_y": float(com[1]),
            "com_z": float(com[2]),
            "com_vx": float(com_velocity[0]),
            "com_vy": float(com_velocity[1]),
            "com_vz": float(com_velocity[2]),
            "kinetic_energy": float(self.data.energy[1]),
            "gravitational_energy": float(self.data.energy[0]),
            "cable_work": float(self._cable_work),
            "damping_work": float(self._damping_work),
            "contact_work": float(self._contact_work),
            "steps_descended": float(self._max_step),
            "max_penetration": float(self._max_penetration),
            "nonadjacent_self_contacts": float(self._nonadjacent_self_contacts),
            "stair_contact_events": float(self._stair_contact_events),
            "max_nonadjacent_self_contacts": float(self._max_nonadjacent_self_contacts),
            "max_stair_contacts": float(self._max_stair_contacts),
        }

    def frame(self) -> dict[str, Any]:
        """Return one renderer/API frame in material order."""

        mujoco.mj_forward(self.model, self.data)
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        quaternions = self._render_quaternions()
        contacts: list[list[float]] = []
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            contacts.append([float(contact.pos[0]), float(contact.pos[1]), float(contact.pos[2])])
        frame = Frame(
            time=self.time,
            positions=positions.tolist(),
            quaternions=quaternions.tolist(),
            contacts=contacts,
            metrics=self._metrics(),
        )
        return frame.model_dump(mode="json")

    def summary(self) -> dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)
        metrics = self._metrics()
        return {
            "status": self._status,
            "scenario": self.config.scenario,
            "prepared": self._prepared,
            "released": self._released,
            "time": self.time,
            "duration": self.duration,
            "dt": self.dt,
            "steps": self._steps,
            "segment_count": self._segments,
            "mass": {
                "configured_kg": float(self.config.material.mass),
                "model_kg": self._model_mass,
                "error_kg": self._model_mass - float(self.config.material.mass),
            },
            "metrics": metrics,
            **metrics,
            "prepare_status": self._settle_status,
            "settle_speed": self._settle_speed,
            "settle_damping": self._settle_damping,
            "settle_steps": self._settle_steps,
            "settle_simulated_seconds": self._settle_steps * self.dt,
            "settle_residual_acceleration": None if not math.isfinite(self._settle_residual_acceleration) else self._settle_residual_acceleration,
            "settle_residual_threshold": 1.0,
            "settle_damping_note": "temporary uniform relaxation damping; restored to configured physical damping before release and accepted only when residual acceleration is below the reported threshold",
            "freefall_check": {
                "samples": self._freefall_samples,
                "max_height_error": None if not math.isfinite(self._freefall_error) else self._freefall_error,
                "release_com_z": self._release_com_z,
                "release_com_vz": self._release_com_vz,
                "status": "observed" if self._freefall_samples else "not_observed",
            },
            "energy_diagnostics": {
                "approximate": True,
                "d_energy_includes_cable_elastic": False,
                "cable_work_definition": "signed integral of qfrc_passive dot qvel after removing known joint damping",
                "damping_work_definition": "signed integral of -sum(dof_damping * qvel^2)",
                "contact_work_definition": "constraint generalized power; includes solver work and is diagnostic",
            },
            "contact_diagnostics": {
                "nonadjacent_self_contact_events": self._nonadjacent_self_contacts,
                "stair_contact_events": self._stair_contact_events,
                "max_nonadjacent_self_contacts_per_step": self._max_nonadjacent_self_contacts,
                "max_stair_contacts_per_step": self._max_stair_contacts,
                "steps_descended_definition": "highest lower stair index reached by an observed cable/stair contact; COM crossing alone is not counted",
            },
            "static_equilibrium_length": self._static_equilibrium_length,
            "bottom_onset_time": self._bottom_onset_time,
            "collapse_time": self._collapse_time,
            "step_events": self._step_events,
            "movement_classification": self._movement_classification(),
            "research_metric_definitions": {
                "static_equilibrium_length": "vertical distance between material-order end geoms immediately before release or after stairs placement",
                "bottom_onset_time": "first post-release time when the material lower endpoint moves beyond max(0.5 mm, 0.5% initial vertical span)",
                "collapse_time": "first post-release time with endpoint vertical span <= 1.05 * turns * strip_thickness, only when initial span exceeds this threshold",
                "step_events": "new lower stair indices reached by cable/stair contact, with endpoint class from material-order end regions",
            },
        }

    def metadata(self) -> dict[str, Any]:
        static_geoms = [{
            "name": "ground",
            "type": "plane",
            "position": [0.0, 0.0, -100.0 if self.config.scenario == "drop" else 0.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "half_sizes": [100.0, 100.0, 0.01],
        }]
        if self.config.scenario == "stairs":
            for index in range(self.config.scene.step_count):
                height = self.config.scene.step_height * (self.config.scene.step_count - index)
                static_geoms.append({
                    "name": f"stair_{index}",
                    "type": "box",
                    "position": [self.config.scene.step_depth * (index + 0.5), 0.0, height * 0.5],
                    "half_sizes": [self.config.scene.step_depth * 0.5, self.config.scene.step_width * 0.5, height * 0.5],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                })
        return {
            "schema_version": self.config.schema_version,
            "model_version": "helical-box-cable-v1",
            "engine_version": "3.15.0",
            "mujoco_version": getattr(mujoco, "__version__", "3.15.0"),
            "units": {"length": "m", "mass": "kg", "time": "s", "force": "N", "energy": "J", "stiffness": "Pa"},
            "coordinate_system": {"world_up": "+z", "stairs_direction": "+x", "quaternion": "wxyz"},
            "scenario": self.config.scenario,
            "segment_count": self._segments,
            "geom_ids": [int(value) for value in self._geom_ids],
            "body_ids": [int(value) for value in self._body_ids],
            "half_sizes": self._half_sizes.tolist(),
            "section_axes": {"x": "centerline tangent", "y": "strip width", "z": "strip thickness"},
            "collision_policy": "explicit contact exclusions cover only direct adjacent cable bodies; non-adjacent cable geoms retain contype/conaffinity self-collision",
            "steps_descended_definition": "highest lower stair index reached by an observed cable/stair contact; COM crossing alone is not counted",
            "rest_shape": {"curve": "cos(s) sin(s) s", "turns": self.config.material.turns, "height": self._height,
                           "radius": self.config.material.radius, "argument_speed": 2 * self.config.material.turns, "flat": False,
                           "tilt_deg": math.degrees(self._angle)},
            "static_geoms": static_geoms,
            "stairs": [geom for geom in static_geoms if geom["name"].startswith("stair_")],
            "energy_diagnostics": {
                "d_energy_terms": ["gravitational_potential", "joint_spring_potential", "tendon_spring_potential", "flex_edge_potential", "kinetic"],
                "cable_elastic_energy": "not provided by the first-party cable plugin; cable_work is an approximate power integral",
                "contact_energy": "not included in d.energy",
            },
        }
