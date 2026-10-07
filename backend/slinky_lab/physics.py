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

from . import MODEL_VERSION
from .energy import cable_elastic_energy
from .equilibrium import solve_static_equilibrium
from .gait import GaitRecorder
from .initial_conditions import hanging_guess
from .schemas import Contact, Frame, RunConfig


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


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.array([
            0.5 * (matrix[2, 1] - matrix[1, 2]),
            0.5 * (matrix[0, 2] - matrix[2, 0]),
            0.5 * (matrix[1, 0] - matrix[0, 1]),
        ], dtype=float)
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        return np.zeros(3, dtype=float)
    axis = np.array([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ], dtype=float) / (2.0 * sine)
    return axis * angle


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
        # The legacy stairs pose is a rigidly tilted reference helix.  An
        # arched pose must keep the reference model vertical and put all
        # bending into runtime qpos, otherwise qpos0 would encode the hand
        # deformation as a stress-free shape.
        self._angle = (math.radians(config.scene.tilt_deg)
                       if config.scenario == "stairs" and config.scene.initial_pose == "tilted" else 0.0)
        self._offset = self._initial_offset()
        self._anchor = self._cable_endpoint()
        self._rest_vertices, self._rest_frames, self._rest_lengths = self._radial_geometry()
        self.model_xml = self._build_model_xml()
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        self._body_ids, self._geom_ids = self._discover_cable_elements()
        self._endpoint_ids = [self.model.site("S_first").id, self.model.site("S_last").id]
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
        self._gait = GaitRecorder(max(config.material.mass * config.scene.gravity, 1e-8))
        self._settle_speed = 0.0
        self._settle_damping = 0.0
        self._settle_steps = 0
        self._settle_residual_acceleration = float("nan")
        self._settle_force_residual = float("nan")
        self._settle_anchor_error = float("nan")
        self._initial_guess_used = False
        self._static_solve: dict[str, Any] = {"status": "not_run"}
        self._quiet_duration = 0.0
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
        self._qpos0_reference = np.asarray(self.model.qpos0, dtype=float).copy()
        self._initial_pose_target_vertices: np.ndarray | None = None
        self._initial_pose_axis_targets: np.ndarray | None = None
        self._initial_pose_clearance_targets: tuple[float, float] | None = None
        self._initial_pose_end_segments = 0
        self._initial_pose_diagnostics: dict[str, Any] = {}
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

    def _radial_geometry(self) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        material = self.config.material
        theta = np.linspace(0.0, 2.0 * math.pi * material.turns, self._segments + 1)
        local_vertices = np.column_stack((
            material.radius * np.cos(theta),
            material.radius * np.sin(theta),
            self._height * np.linspace(0.0, 1.0, self._segments + 1),
        ))
        # Put the held end at the root of the drop tree. A distal-end hold
        # otherwise creates a dense constraint Jacobian across every hinge.
        if self.config.scenario == "drop":
            local_vertices = local_vertices[::-1].copy()
            theta = theta[::-1].copy()
        rotation = _rotation_y(self._angle)
        vertices = local_vertices @ rotation.T + np.asarray(self._offset, dtype=float)
        frames: list[np.ndarray] = []
        lengths = np.empty(self._segments, dtype=float)
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
            frames.append(np.column_stack((tangent, width, normal)))
            lengths[index] = length
        return vertices, frames, lengths

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
              solref=\"{_xml_float(self.config.numerics.contact_time_constant)} 1\" solimp=\"0.95 0.99 0.001 0.5 2\"/>
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
        # Terrain friction is independent of coil/coil friction. MuJoCo uses
        # the higher-priority geom's contact parameters at a terrain contact.
        for terrain in world.findall("geom"):
            terrain.set("priority", "1")
        instance = ET.SubElement(extension_plugin, "instance", {"name": "rod_material"})
        source_plugin = composite.find("plugin")
        if source_plugin is None:
            raise RuntimeError("cable plugin configuration is missing")
        for config in source_plugin.findall("config"):
            instance.append(ET.fromstring(ET.tostring(config, encoding="unicode")))
        if self.config.scenario == "drop":
            anchor_body = world.find("body[@name='drop_anchor']")
            ET.SubElement(anchor_body, "site", {"name": "hold_site", "size": ".001"})
            hold = root.find("equality/connect")
            hold.attrib.clear()
            hold.attrib.update({"name": "drop_hold", "site1": "S_first", "site2": "hold_site", "solref": "0.001 1",
                                "solimp": "0.99 0.9999 0.0001"})

        material = self.config.material
        vertices = self._rest_vertices
        parent = world
        previous_frame: np.ndarray | None = None
        previous_name: str | None = None
        contact = ET.SubElement(root, "contact")

        for index in range(self._segments):
            length = float(self._rest_lengths[index])
            frame = self._rest_frames[index]
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
                "friction": _vec3((material.self_friction, material.self_friction * 0.8, material.self_friction * 0.02)),
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
        if self.config.scenario == "stairs" and self.config.scene.initial_pose == "arched":
            self._apply_arched_pose()
        root_body = self._body_ids[0]
        joint_start = int(self.model.body_jntadr[root_body])
        joint_count = int(self.model.body_jntnum[root_body])
        if joint_count:
            joint_id = joint_start
            dof_start = int(self.model.jnt_dofadr[joint_id])
            dof_count = self._joint_dof_count(joint_id)
            if dof_count >= 6:
                self.data.qvel[dof_start] = self.config.scene.initial_forward_velocity
                self.data.qvel[dof_start + 1] = self.config.scene.initial_lateral_velocity
                # Free-joint angular qvel uses the root material frame. The
                # scene perturbation is about world +Y, the stair flip axis.
                world_omega = np.array([0.0, self.config.scene.initial_angular_velocity, 0.0])
                self.data.qvel[dof_start + 3:dof_start + 6] = self._rest_frames[0].T @ world_omega
        mujoco.mj_forward(self.model, self.data)
        self._collect_initial_pose_diagnostics()

    def _joint_dof_count(self, joint_id: int) -> int:
        joint_type = int(self.model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            return 6
        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            return 3
        return 1

    def _set_pose_qpos(self, vertices: np.ndarray, frames: np.ndarray) -> None:
        if vertices.shape != (self._segments + 1, 3) or frames.shape != (self._segments, 3, 3):
            raise ValueError("initial pose geometry has an invalid shape")
        self.data.qpos[:3] = vertices[0]
        self.data.qpos[3:7] = _matrix_to_quat(frames[0])
        for index, body_id in enumerate(self._body_ids[1:], 1):
            joint_id = int(self.model.body_jntadr[body_id])
            qadr = int(self.model.jnt_qposadr[joint_id])
            rest_relative = self._rest_frames[index - 1].T @ self._rest_frames[index]
            desired_relative = frames[index - 1].T @ frames[index]
            self.data.qpos[qadr:qadr + 4] = _matrix_to_quat(rest_relative.T @ desired_relative)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _fabrik_fixed_endpoints(points: np.ndarray, lengths: np.ndarray, start: int, end: int, tolerance: float) -> np.ndarray:
        result = np.asarray(points, dtype=float).copy()
        segment_lengths = np.asarray(lengths[start:end], dtype=float)
        start_point = result[start].copy()
        end_point = result[end].copy()
        reach = float(np.linalg.norm(end_point - start_point))
        total_length = float(np.sum(segment_lengths))
        if not math.isfinite(reach) or not math.isfinite(total_length) or reach > total_length + tolerance:
            raise ValueError("arched initial pose endpoints are farther apart than the available cable length")
        fallback = end_point - start_point
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm <= 1.0e-14:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            fallback /= fallback_norm
        for _ in range(512):
            result[start] = start_point
            for index in range(start + 1, end + 1):
                delta = result[index] - result[index - 1]
                norm = float(np.linalg.norm(delta))
                direction = fallback if norm <= 1.0e-14 else delta / norm
                result[index] = result[index - 1] + segment_lengths[index - start - 1] * direction
            result[end] = end_point
            for index in range(end - 1, start - 1, -1):
                delta = result[index] - result[index + 1]
                norm = float(np.linalg.norm(delta))
                direction = fallback if norm <= 1.0e-14 else delta / norm
                result[index] = result[index + 1] + segment_lengths[index - start] * direction
            result[start] = start_point
            errors = np.abs(np.linalg.norm(np.diff(result[start:end + 1], axis=0), axis=1) - segment_lengths)
            if float(np.max(errors, initial=0.0)) <= tolerance:
                return result
        raise ValueError("arched initial pose fixed-length solve did not converge")

    def _apply_arched_pose(self) -> None:
        scene = self.config.scene
        material = self.config.material
        segments_per_turn = int(self._resolved["segments_per_turn"])
        end_segments = int(scene.arch_end_turns) * segments_per_turn
        if self._segments <= 2 * end_segments + 1:
            raise ValueError("arched initial pose needs a non-empty middle cable region")
        half_span = 0.5 * float(scene.step_depth)
        if half_span <= float(material.radius) + 0.5 * float(material.strip_width):
            raise ValueError("arched initial pose is too tight for the coil radius and strip width")
        rise = float(scene.arch_rise if scene.arch_rise is not None else scene.step_depth * 0.5)
        first_top = float(scene.step_height * scene.step_count)
        last_top = float(scene.step_height * (scene.step_count - 1))
        first_clearance_target = 0.0002
        last_clearance_target = float(scene.arch_free_clearance)
        x0 = float(scene.step_depth) * 0.5 + float(scene.launch_offset)
        x1 = x0 + float(scene.step_depth)
        flip = _rotation_y(math.pi)
        reference = np.asarray(self._rest_vertices, dtype=float)
        half_sizes = np.asarray(self._half_sizes, dtype=float)

        def terminal_bottom_offset(indices: range, transform: np.ndarray, endpoint: np.ndarray) -> float:
            values = []
            for index in indices:
                frame = transform @ self._rest_frames[index]
                center = transform @ (reference[index] - endpoint) + frame[:, 0] * (self._rest_lengths[index] * 0.5)
                extent = float(np.sum(np.abs(frame[2, :]) * half_sizes[index]))
                values.append(float(center[2] - extent))
            return float(min(values))

        first_bottom_offset = terminal_bottom_offset(range(0, end_segments), np.eye(3), reference[0])
        last_bottom_offset = terminal_bottom_offset(range(self._segments - end_segments, self._segments), flip, reference[-1])
        first_endpoint_z = first_top + first_clearance_target - first_bottom_offset
        last_endpoint_z = last_top + last_clearance_target - last_bottom_offset
        first_axis_target = np.array([x0, 0.0, first_endpoint_z], dtype=float)
        last_axis_target = np.array([x1, 0.0, last_endpoint_z], dtype=float)
        # The first reference node is one radius along +N from the coil axis;
        # after the 180-degree tail turn the last node is one radius along -N.
        # Targeting the axis centres keeps complete terminal turns over the
        # stair treads instead of placing only a material endpoint at centre.
        first_target = first_axis_target + np.array([material.radius, 0.0, 0.0], dtype=float)
        last_target = last_axis_target - np.array([material.radius, 0.0, 0.0], dtype=float)
        target = np.empty_like(reference)
        target[:end_segments + 1] = first_target + (reference[:end_segments + 1] - reference[0])
        target[self._segments - end_segments:] = last_target + (reference[self._segments - end_segments:] - reference[-1]) @ flip.T
        middle_start_z = first_endpoint_z + float(reference[end_segments, 2] - reference[0, 2])
        middle_end_z = last_endpoint_z + float((flip @ (reference[self._segments - end_segments] - reference[-1]))[2])

        def axis_frame(value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            value = float(np.clip(value, 0.0, 1.0))
            x = x0 + half_span * (1.0 - math.cos(math.pi * value))
            z = middle_start_z + (middle_end_z - middle_start_z) * (3.0 * value**2 - 2.0 * value**3) + rise * math.sin(math.pi * value)
            tangent = np.array([
                half_span * math.pi * math.sin(math.pi * value),
                0.0,
                (middle_end_z - middle_start_z) * 6.0 * value * (1.0 - value) + rise * math.pi * math.cos(math.pi * value),
            ], dtype=float)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1.0e-14:
                raise ValueError("arched initial pose has a zero arch-axis tangent")
            tangent /= tangent_norm
            binormal = np.array([0.0, 1.0, 0.0], dtype=float)
            radial = np.cross(binormal, tangent)
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm <= 1.0e-14:
                raise ValueError("arched initial pose cannot construct a radial frame")
            radial /= radial_norm
            return np.array([x, 0.0, z], dtype=float), tangent, radial

        theta = np.linspace(0.0, 2.0 * math.pi * material.turns, self._segments + 1)
        middle_start = end_segments
        middle_end = self._segments - end_segments
        middle_span = float(middle_end - middle_start)
        for index in range(middle_start + 1, middle_end):
            value = (index - middle_start) / middle_span
            axis, _, radial = axis_frame(value)
            target[index] = axis + material.radius * (math.cos(theta[index]) * radial + math.sin(theta[index]) * np.array([0.0, 1.0, 0.0]))
        length_tolerance = max(1.0e-10, float(np.min(self._rest_lengths)) * 1.0e-7)
        target = self._fabrik_fixed_endpoints(target, self._rest_lengths, middle_start, middle_end, length_tolerance)

        frames = np.empty((self._segments, 3, 3), dtype=float)
        for index in range(self._segments):
            if index < end_segments:
                frames[index] = self._rest_frames[index]
                continue
            if index >= self._segments - end_segments:
                frames[index] = flip @ self._rest_frames[index]
                continue
            tangent = target[index + 1] - target[index]
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1.0e-14:
                raise ValueError("arched initial pose produced a zero-length middle segment")
            tangent /= tangent_norm
            value = (index + 0.5 - middle_start) / middle_span
            _, _, radial = axis_frame(value)
            width = -(math.cos(0.5 * (theta[index] + theta[index + 1])) * radial + math.sin(0.5 * (theta[index] + theta[index + 1])) * np.array([0.0, 1.0, 0.0]))
            width -= tangent * float(np.dot(width, tangent))
            width_norm = float(np.linalg.norm(width))
            if width_norm <= 1.0e-14:
                raise ValueError("arched initial pose produced a degenerate section frame")
            width /= width_norm
            normal = np.cross(tangent, width)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1.0e-14:
                raise ValueError("arched initial pose produced a degenerate normal frame")
            normal /= normal_norm
            width = np.cross(normal, tangent)
            frames[index] = np.column_stack((tangent, width, normal))

        self._initial_pose_target_vertices = target.copy()
        self._initial_pose_axis_targets = np.vstack((first_axis_target, last_axis_target))
        self._initial_pose_clearance_targets = (first_clearance_target, last_clearance_target)
        self._initial_pose_end_segments = end_segments
        self._set_pose_qpos(target, frames)
        self.data.qvel[:] = 0.0
        actual_body_nodes = np.asarray(self.data.xpos, dtype=float)[self._body_ids]
        actual_last_node = np.asarray(self.data.site_xpos, dtype=float)[self._endpoint_ids[1]][None, :]
        actual_nodes = np.concatenate((actual_body_nodes, actual_last_node), axis=0)
        actual_link_errors = np.abs(np.linalg.norm(np.diff(actual_nodes, axis=0), axis=1) - self._rest_lengths)
        fk_tolerance = max(1.0e-8, float(np.min(self._rest_lengths)) * 1.0e-5)
        if float(np.max(actual_link_errors, initial=0.0)) > fk_tolerance:
            raise ValueError("arched initial pose FK changed a cable segment length beyond tolerance")
        if float(np.max(np.linalg.norm(actual_nodes - target, axis=1), initial=0.0)) > fk_tolerance:
            raise ValueError("arched initial pose FK did not realize the requested nodes within tolerance")
        matrices = np.asarray(self.data.geom_xmat[self._geom_ids], dtype=float).reshape(-1, 3, 3)
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        half_sizes = np.asarray(self._half_sizes, dtype=float)
        z_extent = np.sum(np.abs(matrices[:, 2, :]) * half_sizes, axis=1)
        x_extent = np.sum(np.abs(matrices[:, 0, :]) * half_sizes, axis=1)
        bottoms = positions[:, 2] - z_extent
        x_mins = positions[:, 0] - x_extent
        x_maxs = positions[:, 0] + x_extent
        first_slice = slice(0, end_segments)
        last_slice = slice(self._segments - end_segments, self._segments)
        first_clearance = float(np.min(bottoms[first_slice]) - first_top)
        last_clearance = float(np.min(bottoms[last_slice]) - last_top)
        first_x_range = (float(np.min(x_mins[first_slice])), float(np.max(x_maxs[first_slice])))
        last_x_range = (float(np.min(x_mins[last_slice])), float(np.max(x_maxs[last_slice])))
        if first_clearance < first_clearance_target - fk_tolerance or last_clearance < last_clearance_target - fk_tolerance:
            raise ValueError("arched initial pose terminal turns penetrate a stair tread or leave no free-end clearance")
        if first_x_range[0] < -fk_tolerance or first_x_range[1] > float(scene.step_depth) + fk_tolerance:
            raise ValueError("arched initial pose first terminal turn does not fit stair_0")
        if last_x_range[0] < float(scene.step_depth) - fk_tolerance or last_x_range[1] > 2.0 * float(scene.step_depth) + fk_tolerance:
            raise ValueError("arched initial pose last terminal turn does not fit stair_1")

    def _collect_initial_pose_diagnostics(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        actual_body_nodes = np.asarray(self.data.xpos, dtype=float)[self._body_ids]
        actual_last_node = np.asarray(self.data.site_xpos, dtype=float)[self._endpoint_ids[1]][None, :]
        actual_nodes = np.concatenate((actual_body_nodes, actual_last_node), axis=0)
        actual_links = np.linalg.norm(np.diff(actual_nodes, axis=0), axis=1)
        link_errors = actual_links - self._rest_lengths
        endpoint_actual = np.asarray(self.data.site_xpos[self._endpoint_ids], dtype=float)
        target = self._initial_pose_target_vertices
        target_nodes = [] if target is None else target.tolist()
        endpoint_target = [] if target is None else target[[0, -1]].tolist()
        endpoint_errors = [] if target is None else np.linalg.norm(endpoint_actual - target[[0, -1]], axis=1).tolist()
        matrices = np.asarray(self.data.geom_xmat[self._geom_ids], dtype=float).reshape(-1, 3, 3)
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        half_sizes = np.asarray(self._half_sizes, dtype=float)
        z_extent = np.sum(np.abs(matrices[:, 2, :]) * half_sizes, axis=1)
        x_extent = np.sum(np.abs(matrices[:, 0, :]) * half_sizes, axis=1)
        bottoms = positions[:, 2] - z_extent
        x_mins = positions[:, 0] - x_extent
        x_maxs = positions[:, 0] + x_extent
        end_count = self._initial_pose_end_segments
        if end_count:
            first_slice = slice(0, end_count)
            last_slice = slice(self._segments - end_count, self._segments)
            first_clearance = float(np.min(bottoms[first_slice]) - self.config.scene.step_height * self.config.scene.step_count)
            last_clearance = float(np.min(bottoms[last_slice]) - self.config.scene.step_height * (self.config.scene.step_count - 1))
            first_x_range = [float(np.min(x_mins[first_slice])), float(np.max(x_maxs[first_slice]))]
            last_x_range = [float(np.min(x_mins[last_slice])), float(np.max(x_maxs[last_slice]))]
        else:
            first_clearance = None
            last_clearance = None
            first_x_range = None
            last_x_range = None
        stair_contact_count = 0
        self_contact_count = 0
        initial_penetration = 0.0
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            initial_penetration = max(initial_penetration, max(0.0, -float(contact.dist)))
            if int(contact.geom[0]) in self._stair_geom_ids or int(contact.geom[1]) in self._stair_geom_ids:
                stair_contact_count += 1
            elif int(contact.geom[0]) in self._cable_geom_to_material and int(contact.geom[1]) in self._cable_geom_to_material:
                self_contact_count += 1
        passive_force = np.asarray(self.data.qfrc_passive, dtype=float)
        acceleration = np.asarray(self.data.qacc, dtype=float)
        resolved = "arched" if target is not None else ("tilted" if self.config.scenario == "stairs" else "drop_reference")
        equilibrium_status = "manual_non_equilibrium" if target is not None else ("drop_hold_pending" if self.config.scenario == "drop" else "reference_pose")
        axis_targets = [] if self._initial_pose_axis_targets is None else self._initial_pose_axis_targets.tolist()
        clearance_targets = [] if self._initial_pose_clearance_targets is None else list(self._initial_pose_clearance_targets)
        self._initial_pose_diagnostics = {
            "requested": self.config.scene.initial_pose,
            "resolved": resolved,
            "target_stair_indices": [0, 1] if target is not None else [],
            "target_axis_centres": axis_targets,
            "target_clearances_m": clearance_targets,
            "target_nodes": target_nodes,
            "actual_nodes": actual_nodes.tolist() if target is not None else [],
            "target_endpoints": endpoint_target,
            "actual_endpoints": endpoint_actual.tolist(),
            "endpoint_errors_m": endpoint_errors,
            "max_endpoint_error_m": float(max(endpoint_errors, default=0.0)),
            "max_link_error_m": float(np.max(np.abs(link_errors), initial=0.0)),
            "rms_link_error_m": float(np.sqrt(np.mean(link_errors**2))) if link_errors.size else 0.0,
            "qpos0_unchanged": bool(np.array_equal(np.asarray(self.model.qpos0, dtype=float), self._qpos0_reference)),
            "initial_elastic_energy_estimate": float(self._elastic_energy_estimate()),
            "initial_passive_force_max": float(np.max(np.abs(passive_force), initial=0.0)),
            "initial_acceleration_max": float(np.max(np.abs(acceleration), initial=0.0)),
            "initial_contact_count": int(self.data.ncon),
            "initial_stair_contact_count": stair_contact_count,
            "initial_self_contact_count": self_contact_count,
            "initial_max_penetration_m": initial_penetration,
            "actual_first_clearance_m": first_clearance,
            "actual_free_end_clearance_m": last_clearance,
            "first_end_x_range_m": first_x_range,
            "last_end_x_range_m": last_x_range,
            "first_end_within_stair0": bool(first_x_range is not None and first_x_range[0] >= -1.0e-6 and first_x_range[1] <= self.config.scene.step_depth + 1.0e-6),
            "last_end_within_stair1": bool(last_x_range is not None and last_x_range[0] >= self.config.scene.step_depth - 1.0e-6 and last_x_range[1] <= 2.0 * self.config.scene.step_depth + 1.0e-6),
            "equilibrium_status": equilibrium_status,
            "equality_active_count": int(np.count_nonzero(np.asarray(self.data.eq_active, dtype=int))),
            "actuator_count": int(self.model.nu),
            "applied_force_max": float(np.max(np.abs(np.asarray(self.data.qfrc_applied, dtype=float)), initial=0.0)),
            "release_driving_note": "initial pose is written to data.qpos; no persistent equality, actuator, or applied force is used",
        }

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
            self._apply_hanging_guess()
            self._report_progress(progress, "static_solve", 0.0)
            self._static_solve = solve_static_equilibrium(
                self.model, self.data, self._anchor, endpoint_site_id=self._endpoint_ids[0],
                should_cancel=should_cancel, max_calls=max(10000, self._segments * 150),
                time_limit_s=min(60.0, self.config.numerics.max_wall_seconds * 0.5),
                length_scale=self.config.material.radius,
            )
            if self._static_solve["status"] == "cancelled" or self._is_cancelled(should_cancel):
                self._status = "cancelled"
                return
            self._check_equilibrium()
            physical_damping = self.model.dof_damping.copy()
            material = self.config.material
            inertia = float(np.max(self.model.body_inertia[self._body_ids]))
            moment = max(material.strip_width * material.strip_thickness**3,
                         material.strip_thickness * material.strip_width**3) / 12
            rotational_stiffness = material.young_modulus * moment / float(np.min(self._rest_lengths))
            # Relaxation damping follows sqrt(inertia * stiffness), so it does
            # not freeze microscopic hinges with a macroscopic damping value.
            self._settle_damping = 6 * math.sqrt(inertia * rotational_stiffness)
            self.model.dof_damping[6:] = np.maximum(physical_damping[6:], self._settle_damping)
            self.model.dof_damping[:3] = material.mass * 10
            self.model.dof_damping[3:6] = material.mass * material.radius**2 * 10
            self._settle_status = "not_converged"
            try:
                check_interval = max(1, int(0.05 / self.dt))
                for index in range(0 if self._equilibrium_converged() else settle_steps):
                    if self._is_cancelled(should_cancel):
                        self._status = "cancelled"
                        return
                    previous_time = self.time
                    mujoco.mj_step(self.model, self.data)
                    self._check_dynamics(previous_time)
                    self._settle_steps = index + 1
                    if (index + 1) % check_interval == 0 or index + 1 == settle_steps:
                        self._check_equilibrium()
                        self._report_progress(progress, "settle", (index + 1) / settle_steps)
                        if self.time >= 0.2 and self._equilibrium_converged():
                            self._settle_status = "converged"
                            break
            finally:
                self.model.dof_damping[:] = physical_damping
            self._check_equilibrium()
            if self._equilibrium_converged():
                self._settle_status = "converged"
                self.data.qvel[:] = 0
            # Residuals above were measured with the hold still active. The
            # post-release acceleration must include gravity and is not a
            # static equilibrium residual.
            self.data.eq_active[0] = 0
            self.data.time = 0.0
            mujoco.mj_forward(self.model, self.data)
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

    def _apply_hanging_guess(self) -> None:
        guess = hanging_guess(self.config, self._rest_lengths, self._anchor)
        if guess is None:
            return
        vertices, frames = guess
        vertices = vertices[::-1].copy()
        frames = frames[::-1].copy() @ np.diag([-1.0, 1.0, -1.0])
        self._set_pose_qpos(vertices, frames)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._initial_guess_used = True

    def _check_equilibrium(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_energyVel(self.model, self.data)
        self._settle_speed = math.sqrt(max(0.0, 2 * float(self.data.energy[1]) / self._model_mass))
        velocity = self.data.qvel.copy()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._settle_residual_acceleration = float(np.max(np.abs(self.data.qacc)))
        residual = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, residual, self.data.qacc)
        force_scale = max(self._model_mass * self.config.scene.gravity, 1e-8)
        torque_scale = force_scale * self.config.material.radius
        self._settle_force_residual = max(float(np.max(np.abs(residual[:3]))) / force_scale,
                                          float(np.max(np.abs(residual[3:]))) / torque_scale)
        self._settle_anchor_error = float(np.linalg.norm(self.data.site_xpos[self._endpoint_ids[0]] - self._anchor))
        self.data.qvel[:] = velocity
        mujoco.mj_forward(self.model, self.data)

    def _equilibrium_converged(self) -> bool:
        return self._settle_speed <= 0.002 and self._settle_force_residual <= 0.01 and self._settle_anchor_error <= 0.0002

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
        support_forces: dict[int, dict[str, float]] = {}
        contact_force = np.zeros(6)
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
                end_fraction = int(self._resolved["segments_per_turn"])
                if material_index < end_fraction:
                    label = "first"
                elif material_index >= len(self._geom_ids) - end_fraction:
                    label = "last"
                else:
                    label = "middle"
                self._last_stair_contact_labels.setdefault(step_index, set()).add(label)
                if step_index > 0 and self._is_tread_contact(contact, step_index):
                    mujoco.mj_contactForce(self.model, self.data, index, contact_force)
                    forces = support_forces.setdefault(step_index, {"first": 0.0, "last": 0.0, "middle": 0.0})
                    forces[label] += max(0.0, float(contact_force[0]))
        self._nonadjacent_self_contacts += nonadjacent
        self._stair_contact_events += stair_contacts
        self._max_nonadjacent_self_contacts = max(self._max_nonadjacent_self_contacts, nonadjacent)
        self._max_stair_contacts = max(self._max_stair_contacts, stair_contacts)
        if self.config.scenario == "stairs":
            self._gait.update(self.time, self.dt, support_forces)

    def _is_tread_contact(self, contact: Any, step: int) -> bool:
        scene = self.config.scene
        top = scene.step_height * (scene.step_count - step)
        point = np.asarray(contact.pos)
        tolerance = max(0.0005, self.config.material.strip_thickness * 2)
        return (abs(float(contact.frame[2])) >= 0.8 and abs(float(point[2]) - top) <= tolerance
                and scene.step_depth * step - tolerance <= point[0] <= scene.step_depth * (step + 1) + tolerance
                and abs(float(point[1])) <= scene.step_width * 0.5 + tolerance)

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
        positions = np.asarray(self.data.site_xpos[self._endpoint_ids], dtype=float)
        return float(abs(positions[-1, 2] - positions[0, 2]))

    def _prepare_research_metrics(self) -> None:
        positions = np.asarray(self.data.site_xpos[self._endpoint_ids], dtype=float)
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
        positions = np.asarray(self.data.site_xpos[self._endpoint_ids], dtype=float)
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
        if self.config.scenario == "stairs":
            if abs(self._com_position()[1]) > self.config.scene.step_width * 0.5 + self.config.material.radius:
                return "side_fall"
            if self._quiet_duration >= 0.1:
                return "stopped"
        support = self._gait.summary()
        if support["confirmed_flip_count"] >= 2:
            return "flip"
        if support["candidate_flip_count"] >= 2:
            return "ambiguous_flip"
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
        previous_time = self.time
        mujoco.mj_step(self.model, self.data)
        self._check_dynamics(previous_time)
        if self.duration - 1e-12 <= self.time < self.duration:
            self.data.time = self.duration
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
        if self.config.scenario == "stairs":
            mujoco.mj_energyVel(self.model, self.data)
            speed = math.sqrt(max(0.0, 2 * float(self.data.energy[1]) / self._model_mass))
            self._quiet_duration = self._quiet_duration + self.dt if speed < 0.005 else 0.0
        if self.time >= self.duration:
            self._status = "complete"
        else:
            self._status = "running"

    def _check_dynamics(self, previous_time: float) -> None:
        warnings = [mujoco.mjtWarning(index).name for index, item in enumerate(self.data.warning) if item.number]
        if warnings or self.time <= previous_time:
            self._status = "unstable"
            raise RuntimeError(f"MuJoCo rejected the integration step: {', '.join(warnings) or 'time reset'}")

    def _render_quaternions(self) -> np.ndarray:
        matrices = np.asarray(self.data.geom_xmat[self._geom_ids], dtype=float).reshape(-1, 3, 3)
        quaternions = np.vstack([_matrix_to_quat(matrix) for matrix in matrices])
        if self._previous_quaternions is not None:
            signs = np.sum(quaternions * self._previous_quaternions, axis=1) < 0.0
            quaternions[signs] *= -1.0
        self._previous_quaternions = quaternions.copy()
        return quaternions

    def _elastic_energy_estimate(self) -> float:
        material = self.config.material
        return cable_elastic_energy(self.model, self.data, self._body_ids, material.young_modulus, material.shear_modulus)

    def _metrics(self) -> dict[str, float]:
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        matrices = np.asarray(self.data.geom_xmat[self._geom_ids], dtype=float).reshape(-1, 3, 3)
        extents = np.sum(np.abs(matrices[:, 2, :]) * self._half_sizes, axis=1)
        com = self._com_position()
        com_velocity = self._com_velocity()
        spatial_top_z = float(np.max(positions[:, 2] + extents))
        spatial_bottom_z = float(np.min(positions[:, 2] - extents))
        top_endpoint_index = self._top_endpoint_index
        bottom_endpoint_index = self._bottom_endpoint_index
        if self._static_equilibrium_length is None:
            top_endpoint_index = 0 if positions[0, 2] >= positions[-1, 2] else -1
            bottom_endpoint_index = -1 if top_endpoint_index == 0 else 0
        material_top_z = float(positions[top_endpoint_index, 2])
        material_bottom_z = float(positions[bottom_endpoint_index, 2])
        endpoints = np.asarray(self.data.site_xpos[self._endpoint_ids], dtype=float)
        material_top_z = float(endpoints[top_endpoint_index, 2])
        material_bottom_z = float(endpoints[bottom_endpoint_index, 2])
        top_z = material_top_z
        bottom_z = material_bottom_z
        mujoco.mj_energyPos(self.model, self.data)
        mujoco.mj_energyVel(self.model, self.data)
        elastic_energy = self._elastic_energy_estimate()
        return {
            "top_z": top_z,
            "bottom_z": bottom_z,
            "spatial_top_z": spatial_top_z,
            "spatial_bottom_z": spatial_bottom_z,
            "material_top_z": material_top_z,
            "material_bottom_z": material_bottom_z,
            "material_first_z": float(endpoints[0, 2]),
            "material_last_z": float(endpoints[-1, 2]),
            "material_vertical_span": self._material_vertical_span(),
            "com_x": float(com[0]),
            "com_y": float(com[1]),
            "com_z": float(com[2]),
            "com_vx": float(com_velocity[0]),
            "com_vy": float(com_velocity[1]),
            "com_vz": float(com_velocity[2]),
            "kinetic_energy": float(self.data.energy[1]),
            "gravitational_energy": float(self.data.energy[0]),
            "elastic_energy_estimate": elastic_energy,
            "cable_work": float(self._cable_work),
            "damping_work": float(self._damping_work),
            "contact_work": float(self._contact_work),
            "steps_descended": float(self._max_step),
            "max_penetration": float(self._max_penetration),
            "nonadjacent_self_contacts": float(self._nonadjacent_self_contacts),
            "stair_contact_events": float(self._stair_contact_events),
            "max_nonadjacent_self_contacts": float(self._max_nonadjacent_self_contacts),
            "max_stair_contacts": float(self._max_stair_contacts),
            "confirmed_flip_count": float(self._gait.summary()["confirmed_flip_count"]),
        }

    def frame(self) -> dict[str, Any]:
        """Return one renderer/API frame in material order."""

        mujoco.mj_forward(self.model, self.data)
        positions = np.asarray(self.data.geom_xpos[self._geom_ids], dtype=float)
        quaternions = self._render_quaternions()
        contacts: list[list[float]] = []
        contact_details: list[Contact] = []
        force = np.zeros(6)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            position = [float(value) for value in contact.pos]
            contacts.append(position)
            a, b = int(contact.geom[0]), int(contact.geom[1])
            material_a = self._cable_geom_to_material.get(a)
            material_b = self._cable_geom_to_material.get(b)
            stair_geom = a if a in self._stair_geom_ids else (b if b in self._stair_geom_ids else None)
            step = self._stair_geom_to_step.get(stair_geom)
            material_index = material_a if material_a is not None else material_b
            kind = "self" if material_a is not None and material_b is not None else ("stair" if step is not None else "external")
            mujoco.mj_contactForce(self.model, self.data, index, force)
            contact_details.append(Contact(position=position, geom_a=a, geom_b=b, normal_force=max(0.0, float(force[0])),
                                           kind=kind, stair_step=step, material_index=material_index,
                                           surface="tread" if step is not None and self._is_tread_contact(contact, step) else "other"))
        frame = Frame(
            time=self.time,
            positions=positions.tolist(),
            quaternions=quaternions.tolist(),
            contacts=contacts,
            contact_details=contact_details,
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
            "initial_guess_used": self._initial_guess_used,
            "static_solve": self._static_solve,
            "settle_simulated_seconds": self._settle_steps * self.dt,
            "settle_residual_acceleration": None if not math.isfinite(self._settle_residual_acceleration) else self._settle_residual_acceleration,
            "settle_force_residual": None if not math.isfinite(self._settle_force_residual) else self._settle_force_residual,
            "settle_anchor_error_m": None if not math.isfinite(self._settle_anchor_error) else self._settle_anchor_error,
            "settle_thresholds": {"equivalent_speed_m_s": 0.002, "relative_force_residual": 0.01, "anchor_error_m": 0.0002},
            "settle_damping_note": "temporary inertia-scaled relaxation damping; physical damping restored before release, residual measured while held",
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
                "elastic_energy_estimate": metrics["elastic_energy_estimate"],
                "elastic_energy_definition": "plugin rotation-vector difference and rectangular Saint-Venant J, Iy, Iz; diagnostic only",
                "elastic_energy_gradient_check": "small joint rotation differs from plugin torque by about 11%; do not treat as an exact potential",
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
            "support_diagnostics": self._gait.summary(),
            "initial_pose_diagnostics": self._initial_pose_diagnostics,
            "quiet_duration_s": self._quiet_duration,
            "research_metric_definitions": {
                "static_equilibrium_length": "vertical distance between material endpoints immediately before release or after stairs placement",
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
            "model_version": MODEL_VERSION,
            "engine_version": "3.15.0",
            "mujoco_version": getattr(mujoco, "__version__", "3.15.0"),
            "units": {"length": "m", "mass": "kg", "time": "s", "force": "N", "energy": "J", "stiffness": "Pa"},
            "coordinate_system": {"world_up": "+z", "stairs_direction": "+x", "quaternion": "wxyz"},
            "initial_velocity_axes": {"forward": "world +x", "lateral": "world +y", "angular": "world +y"},
            "scenario": self.config.scenario,
            "segment_count": self._segments,
            "geom_ids": [int(value) for value in self._geom_ids],
            "body_ids": [int(value) for value in self._body_ids],
            "half_sizes": self._half_sizes.tolist(),
            "section_axes": {"x": "centerline tangent", "y": "strip width", "z": "strip thickness"},
            "friction": {"self": self.config.material.self_friction, "environment": self.config.material.friction,
                         "combination": "terrain priority 1 selects environment friction; cable/cable uses self friction"},
            "contact_solver": {"time_constant_s": self.config.numerics.contact_time_constant,
                               "damping_ratio": 1.0, "solimp": [0.95, 0.99, 0.001, 0.5, 2.0],
                               "note": "MuJoCo regularization; effective time constant is clipped to at least twice the timestep"},
            "collision_policy": "explicit contact exclusions cover only direct adjacent cable bodies; non-adjacent cable geoms retain contype/conaffinity self-collision",
            "steps_descended_definition": "highest lower stair index reached by an observed cable/stair contact; COM crossing alone is not counted",
            "rest_shape": {"curve": "cos(s) sin(s) s", "turns": self.config.material.turns, "height": self._height,
                           "radius": self.config.material.radius, "argument_speed": 2 * self.config.material.turns, "flat": False,
                           "tilt_deg": math.degrees(self._angle)},
            "initial_pose_diagnostics": self._initial_pose_diagnostics,
            "static_geoms": static_geoms,
            "stairs": [geom for geom in static_geoms if geom["name"].startswith("stair_")],
            "energy_diagnostics": {
                "d_energy_terms": ["gravitational_potential", "joint_spring_potential", "tendon_spring_potential", "flex_edge_potential", "kinetic"],
                "cable_elastic_energy": "not provided by the first-party cable plugin; cable_work is an approximate power integral",
                "elastic_energy_estimate": "available as an independent rectangular-section frame diagnostic; not included in d.energy",
                "contact_energy": "not included in d.energy",
            },
        }
