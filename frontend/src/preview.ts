import * as THREE from "three";
import type { Frame, RenderMetadata, RunConfig } from "./types";

const TAU = Math.PI * 2;

export function makePreviewFrame(config: RunConfig, time = 0): Frame {
  const segments = Math.max(24, Math.min(180, config.material.turns * (config.numerics.segments_per_turn || 12)));
  const radius = config.material.radius;
  const height = Math.max(config.material.pitch * config.material.turns, config.material.strip_thickness * 2);
  const positions: number[][] = [];
  const quaternions: number[][] = [];
  const angle = config.scenario === "stairs" ? (config.scene.tilt_deg * Math.PI) / 180 : 0;
  const rotation = new THREE.Matrix4().makeRotationY(angle);
  const localVertices: THREE.Vector3[] = [];
  for (let index = 0; index <= segments; index += 1) {
    const u = index / segments;
    localVertices.push(new THREE.Vector3(Math.cos(u * config.material.turns * TAU) * radius, Math.sin(u * config.material.turns * TAU) * radius, height * u).applyMatrix4(rotation));
  }
  const minLocalZ = Math.min(...localVertices.map((point) => point.z));
  const offset = config.scenario === "drop"
    ? new THREE.Vector3(0, 0, Math.max(0.28, config.scene.step_height * config.scene.step_count + 0.12))
    : new THREE.Vector3(config.scene.step_depth * 0.5 + config.scene.launch_offset, 0, config.scene.step_height * config.scene.step_count - minLocalZ + config.material.strip_thickness);

  for (let index = 0; index < segments; index += 1) {
    const start = localVertices[index];
    const end = localVertices[index + 1];
    const tangent = end.clone().sub(start).normalize();
    const center = start.clone().add(end).multiplyScalar(0.5).add(offset);
    positions.push([center.x, center.y, center.z]);
    // Preview poses use the same centerline tangent convention as the box cable model.
    const width = new THREE.Vector3(-Math.cos((index + 0.5) / segments * config.material.turns * TAU), -Math.sin((index + 0.5) / segments * config.material.turns * TAU), 0).applyMatrix4(rotation).normalize();
    const normal = tangent.clone().cross(width).normalize();
    const correctedWidth = normal.clone().cross(tangent).normalize();
    const basis = new THREE.Matrix4().makeBasis(tangent, correctedWidth, normal);
    const quaternion = new THREE.Quaternion().setFromRotationMatrix(basis);
    quaternions.push([quaternion.w, quaternion.x, quaternion.y, quaternion.z]);
  }

  return {
    time,
    positions,
    quaternions,
    contacts: [],
    metrics: {
      top_z: positions[0]?.[2] || 0,
      bottom_z: positions[positions.length - 1]?.[2] || 0,
      com_z: positions.reduce((sum, point) => sum + point[2], 0) / positions.length,
    },
  };
}

export function makePreviewMetadata(config: RunConfig): RenderMetadata {
  const segmentCount = Math.max(24, Math.min(180, config.material.turns * (config.numerics.segments_per_turn || 12)));
  const angle = config.scenario === "stairs" ? (config.scene.tilt_deg * Math.PI) / 180 : 0;
  const rotation = new THREE.Matrix4().makeRotationY(angle);
  const height = Math.max(config.material.pitch * config.material.turns, config.material.strip_thickness * 2);
  const vertices = Array.from({ length: segmentCount + 1 }, (_, index) => {
    const u = index / segmentCount;
    return new THREE.Vector3(Math.cos(u * config.material.turns * TAU) * config.material.radius, Math.sin(u * config.material.turns * TAU) * config.material.radius, height * u).applyMatrix4(rotation);
  });
  const halfSizes = vertices.slice(0, -1).map((point, index) => [point.distanceTo(vertices[index + 1]) / 2, config.material.strip_width / 2, config.material.strip_thickness / 2]);
  const metadata: RenderMetadata = {
    half_sizes: halfSizes,
    types: Array.from({ length: segmentCount }, () => "box"),
    static: Array.from({ length: segmentCount }, () => false),
    static_geoms: [],
    model_version: "preview-geometry",
    engine_version: "none",
  };
  if (config.scenario === "stairs") {
    const stairs = Math.max(3, config.scene.step_count);
    for (let index = 0; index < stairs; index += 1) {
      const height = config.scene.step_height * (stairs - index);
      metadata.static_geoms!.push({
        name: `preview_stair_${index}`,
        type: "box",
        position: [(index + 0.5) * config.scene.step_depth, 0, height * 0.5],
        quaternion: [1, 0, 0, 0],
        half_sizes: [config.scene.step_depth * 0.5, config.scene.step_width * 0.5, height * 0.5],
        static: true,
      });
    }
  }
  return metadata;
}
