import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Frame, GeomMetadata, RenderMetadata, RunConfig } from "../types";

interface Props {
  config: RunConfig;
  frame: Frame;
  metadata?: RenderMetadata;
  runKey: string;
  showContacts: boolean;
  showTrajectory: boolean;
  geometryMode: "boxes" | "smooth";
  cameraView: "orbit" | "front" | "side" | "top";
  followCamera: boolean;
  onCameraViewChange: (view: "orbit" | "front" | "side" | "top") => void;
}

const PALETTE = ["#ef476f", "#ff7b59", "#ffd166", "#8bd450", "#06d6a0", "#20a4f3", "#8f7aea", "#d85ed7"];

interface SceneState {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  staticObjects: THREE.Group;
  dynamicObjects: THREE.Group;
  overlays: THREE.Group;
  dynamicMesh?: THREE.InstancedMesh;
  dynamicMode: "boxes" | "smooth";
  trajectory: THREE.Vector3[];
  previousTime: number;
  hasFit: boolean;
  followTarget?: THREE.Vector3;
  resizeObserver: ResizeObserver;
  frameRequest: number;
}

function colorForIndex(index: number, count: number): THREE.Color {
  const normalized = count <= 1 ? 0 : index / (count - 1);
  const scaled = normalized * (PALETTE.length - 1);
  const left = Math.floor(scaled);
  const right = Math.min(PALETTE.length - 1, left + 1);
  return new THREE.Color(PALETTE[left]).lerp(new THREE.Color(PALETTE[right]), scaled - left);
}

function toQuaternion(values: number[] | undefined): THREE.Quaternion {
  if (!values || values.length < 4) return new THREE.Quaternion();
  // MuJoCo/API use wxyz, while Three.js uses xyzw.
  return new THREE.Quaternion(values[1], values[2], values[3], values[0]);
}

function effectiveMetadata(metadata: RenderMetadata | undefined): RenderMetadata | undefined {
  return (metadata?.render as RenderMetadata | undefined) || metadata;
}

function getHalfSize(metadata: RenderMetadata | undefined, index: number, config: RunConfig): THREE.Vector3 {
  const segmentCount = Math.max(24, Math.min(180, Math.round(config.material.turns * (config.numerics.segments_per_turn || 12))));
  const angleStep = (config.material.turns * Math.PI * 2) / segmentCount;
  const axialStep = (config.material.pitch * config.material.turns) / segmentCount;
  const centerlineLength = Math.hypot(2 * config.material.radius * Math.sin(angleStep / 2), axialStep);
  const fallback = [centerlineLength / 2, config.material.strip_width / 2, config.material.strip_thickness / 2];
  const value = metadata?.half_sizes?.[index] || fallback;
  return new THREE.Vector3(value[0] ?? fallback[0], value[1] ?? fallback[1], value[2] ?? fallback[2]);
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(material)) material.forEach((item) => item.dispose());
    else material?.dispose();
  });
}

function clearGroup(group: THREE.Group) {
  while (group.children.length) {
    const child = group.children.pop();
    if (child) disposeObject(child);
  }
}

function makeBox(size: THREE.Vector3, material: THREE.Material): THREE.Mesh {
  return new THREE.Mesh(new THREE.BoxGeometry(size.x * 2, size.y * 2, size.z * 2), material);
}

function smoothRibbon(frame: Frame, config: RunConfig, metadata: RenderMetadata | undefined): THREE.Mesh | null {
  if (frame.positions.length < 2) return null;
  const points = frame.positions.map((point) => new THREE.Vector3(point[0], point[1], point[2]));
  const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
  const render = effectiveMetadata(metadata);
  // The physics geoms are rectangular strips. TubeGeometry makes them look like a hose,
  // so the display surface keeps the actual width/thickness and material frame.
  const targetByTurn = Math.ceil(Math.max(1, config.material.turns) * 64) + 1;
  const targetByControl = Math.ceil(Math.max(1, points.length - 1) * 2) + 1;
  // Display tessellation is independent of the solver mesh. More surface
  // samples remove visible facets without changing any computed position.
  const ringCount = Math.min(4096, Math.max(64, targetByTurn, targetByControl));
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  const vertexPositions = new Float32Array(ringCount * corners.length * 3);
  const vertexColors = new Float32Array(ringCount * corners.length * 3);
  const indices: number[] = [];
  const fallbackHalfWidth = config.material.strip_width / 2;
  const fallbackHalfThickness = config.material.strip_thickness / 2;
  const quaternions = frame.quaternions.map(toQuaternion);
  for (let ring = 0; ring < ringCount; ring += 1) {
    const u = ring / (ringCount - 1);
    const position = curve.getPoint(u);
    const materialPosition = u * (points.length - 1);
    const left = Math.floor(materialPosition);
    const right = Math.min(points.length - 1, left + 1);
    const alpha = materialPosition - left;
    const firstQuaternion = quaternions[Math.min(left, quaternions.length - 1)] || new THREE.Quaternion();
    const nextQuaternion = quaternions[Math.min(right, quaternions.length - 1)] || firstQuaternion;
    const orientation = firstQuaternion.clone().slerp(nextQuaternion, alpha).normalize();
    const tangent = curve.getTangent(u).normalize();
    const localTangent = new THREE.Vector3(1, 0, 0).applyQuaternion(orientation).normalize();
    // Align only the tangent; the interpolated quaternion still supplies section roll.
    orientation.premultiply(new THREE.Quaternion().setFromUnitVectors(localTangent, tangent));
    const widthAxis = new THREE.Vector3(0, 1, 0).applyQuaternion(orientation).normalize();
    const thicknessAxis = new THREE.Vector3(0, 0, 1).applyQuaternion(orientation).normalize();
    const leftSize = getHalfSize(render, Math.min(left, points.length - 1), config);
    const rightSize = getHalfSize(render, Math.min(right, points.length - 1), config);
    const halfWidth = THREE.MathUtils.lerp(leftSize.y || fallbackHalfWidth, rightSize.y || fallbackHalfWidth, alpha);
    const halfThickness = THREE.MathUtils.lerp(leftSize.z || fallbackHalfThickness, rightSize.z || fallbackHalfThickness, alpha);
    const materialIndex = render?.material_indices?.[Math.min(left, points.length - 1)] ?? left;
    const color = render?.colors?.[materialIndex] ? new THREE.Color(render.colors[materialIndex]) : colorForIndex(materialIndex, points.length);
    corners.forEach(([widthSign, thicknessSign], corner) => {
      const vertex = position.clone().addScaledVector(widthAxis, widthSign * halfWidth).addScaledVector(thicknessAxis, thicknessSign * halfThickness);
      const offset = (ring * corners.length + corner) * 3;
      vertexPositions[offset] = vertex.x;
      vertexPositions[offset + 1] = vertex.y;
      vertexPositions[offset + 2] = vertex.z;
      vertexColors[offset] = color.r;
      vertexColors[offset + 1] = color.g;
      vertexColors[offset + 2] = color.b;
    });
  }
  for (let ring = 0; ring < ringCount - 1; ring += 1) {
    for (let side = 0; side < corners.length; side += 1) {
      const nextSide = (side + 1) % corners.length;
      const current = ring * corners.length + side;
      const next = (ring + 1) * corners.length + side;
      const currentNext = ring * corners.length + nextSide;
      const nextNext = (ring + 1) * corners.length + nextSide;
      indices.push(current, next, currentNext, next, nextNext, currentNext);
    }
  }
  indices.push(0, 2, 1, 0, 3, 2);
  const last = (ringCount - 1) * corners.length;
  indices.push(last, last + 1, last + 2, last, last + 2, last + 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertexPositions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(vertexColors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.24, metalness: 0.02, side: THREE.DoubleSide });
  return new THREE.Mesh(geometry, material);
}

function staticDefinitions(metadata: RenderMetadata | undefined): GeomMetadata[] {
  return metadata?.static_geoms || metadata?.geoms || [];
}

function renderStaticGeoms(group: THREE.Group, metadata: RenderMetadata | undefined, config: RunConfig) {
  const render = effectiveMetadata(metadata);
  const definitions = staticDefinitions(render);
  for (const definition of definitions) {
    const halfSizes = definition.half_sizes || definition.half_size || [0.1, 0.1, 0.01];
    const size = new THREE.Vector3(halfSizes[0] || 0.1, halfSizes[1] || 0.1, halfSizes[2] || 0.01);
    const type = definition.type || "box";
    const material = new THREE.MeshStandardMaterial({ color: type === "plane" ? "#273750" : "#344663", roughness: 0.82, metalness: 0.02 });
    const mesh = type === "sphere" ? new THREE.Mesh(new THREE.SphereGeometry(Math.max(size.x, size.y, size.z), 12, 8), material) : makeBox(size, material);
    const position = definition.position || definition.center || [0, 0, 0];
    mesh.position.set(position[0] || 0, position[1] || 0, position[2] || 0);
    mesh.quaternion.copy(toQuaternion(definition.quaternion));
    group.add(mesh);
  }
  if (!definitions.length && config.scenario === "stairs") {
    for (let index = 0; index < config.scene.step_count; index += 1) {
      const material = new THREE.MeshStandardMaterial({ color: "#344663", roughness: 0.82 });
      const mesh = makeBox(new THREE.Vector3(config.scene.step_depth / 2, config.scene.step_width / 2, config.scene.step_height / 2), material);
      mesh.position.set((index + 0.5) * config.scene.step_depth, 0, (index + 0.5) * config.scene.step_height - config.scene.step_height * 0.5);
      group.add(mesh);
    }
  }
}

function fitCamera(state: SceneState, frame: Frame, metadata: RenderMetadata | undefined, config: RunConfig) {
  if (state.hasFit) return;
  const render = effectiveMetadata(metadata);
  const box = new THREE.Box3();
  frame.positions.forEach((point, index) => {
    const size = getHalfSize(render, index, config);
    box.expandByPoint(new THREE.Vector3(point[0] - size.x, point[1] - size.y, point[2] - size.z));
    box.expandByPoint(new THREE.Vector3(point[0] + size.x, point[1] + size.y, point[2] + size.z));
  });
  staticDefinitions(render).forEach((definition) => {
    // The solver keeps a 100 m ground plane at z=-100 for contact stability. It must not
    // participate in framing, otherwise the useful cable bounds become a tiny speck.
    if (definition.type === "plane" || definition.name === "ground") return;
    const halfSizes = definition.half_sizes || definition.half_size || [0.1, 0.1, 0.01];
    const position = definition.position || definition.center || [0, 0, 0];
    box.expandByPoint(new THREE.Vector3(position[0] - halfSizes[0], position[1] - halfSizes[1], position[2] - halfSizes[2]));
    box.expandByPoint(new THREE.Vector3(position[0] + halfSizes[0], position[1] + halfSizes[1], position[2] + halfSizes[2]));
  });
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() * 0.5, 0.05);
  state.controls.target.copy(center);
  state.camera.position.copy(center.clone().add(new THREE.Vector3(0.9, -1.1, 0.75).normalize().multiplyScalar(radius * 3.05)));
  state.camera.near = Math.max(radius / 100, 0.0001);
  state.camera.far = Math.max(radius * 20, 2);
  state.camera.updateProjectionMatrix();
  state.controls.update();
  state.hasFit = true;
}

function frameCenter(frame: Frame): THREE.Vector3 | undefined {
  if (!frame.positions.length) return undefined;
  return frame.positions.reduce((center, point) => center.add(new THREE.Vector3(point[0], point[1], point[2])), new THREE.Vector3()).multiplyScalar(1 / frame.positions.length);
}

export function SceneView({ config, frame, metadata, runKey, showContacts, showTrajectory, geometryMode, cameraView, followCamera, onCameraViewChange }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<SceneState>();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#091321");
    const camera = new THREE.PerspectiveCamera(36, 1, 0.0001, 100);
    camera.up.set(0, 0, 1);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0.1);
    const ambient = new THREE.HemisphereLight("#eaf6ff", "#18243b", 2.1);
    scene.add(ambient);
    const key = new THREE.DirectionalLight("#ffe8d5", 2.35);
    key.position.set(2.4, -3.2, 4.5);
    scene.add(key);
    const fill = new THREE.DirectionalLight("#8fd9ff", 1.05);
    fill.position.set(-2.5, 1.8, 2.6);
    scene.add(fill);
    const rim = new THREE.PointLight("#ff8eaf", 0.75, 4);
    rim.position.set(-1.5, 0.5, 1.5);
    scene.add(rim);
    const grid = new THREE.GridHelper(0.8, 16, "#1c3850", "#142b40");
    // GridHelper is XZ by default; rotate it to the XY floor for a Z-up world.
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(0.11));
    const staticObjects = new THREE.Group();
    const dynamicObjects = new THREE.Group();
    const overlays = new THREE.Group();
    scene.add(staticObjects, dynamicObjects, overlays);
    const resizeObserver = new ResizeObserver(() => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    });
    resizeObserver.observe(mount);
    const state: SceneState = { scene, camera, renderer, controls, staticObjects, dynamicObjects, overlays, dynamicMode: "boxes", trajectory: [], previousTime: -1, hasFit: false, resizeObserver, frameRequest: 0 };
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      state.frameRequest = requestAnimationFrame(loop);
    };
    state.frameRequest = requestAnimationFrame(loop);
    stateRef.current = state;
    setReady(true);
    return () => {
      cancelAnimationFrame(state.frameRequest);
      resizeObserver.disconnect();
      controls.dispose();
      clearGroup(staticObjects);
      clearGroup(dynamicObjects);
      clearGroup(overlays);
      renderer.dispose();
      if (renderer.domElement.parentElement === mount) mount.removeChild(renderer.domElement);
      stateRef.current = undefined;
      setReady(false);
    };
  }, []);

  useEffect(() => {
    const state = stateRef.current;
    if (!state || !ready) return;
    clearGroup(state.staticObjects);
    clearGroup(state.dynamicObjects);
    clearGroup(state.overlays);
    state.dynamicMesh = undefined;
    state.dynamicMode = "boxes";
    state.trajectory = [];
    state.previousTime = -1;
    state.followTarget = undefined;
    state.hasFit = false;
  }, [runKey, ready]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state || !ready) return;
    clearGroup(state.staticObjects);
    renderStaticGeoms(state.staticObjects, metadata, config);
    state.followTarget = undefined;
    state.hasFit = false;
    fitCamera(state, frame, metadata, config);
  }, [metadata, config.scenario, config.scene.step_count, config.scene.step_depth, config.scene.step_height, config.scene.step_width, runKey, ready]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state || !ready) return;
    const render = effectiveMetadata(metadata);
    if (geometryMode === "smooth") {
      if (state.dynamicMode !== "smooth") {
        clearGroup(state.dynamicObjects);
        state.dynamicMesh = undefined;
        state.dynamicMode = "smooth";
      } else {
        clearGroup(state.dynamicObjects);
      }
      const ribbon = smoothRibbon(frame, config, metadata);
      if (ribbon) state.dynamicObjects.add(ribbon);
    } else {
      if (state.dynamicMode !== "boxes" || !state.dynamicMesh || state.dynamicMesh.count !== frame.positions.length) {
        clearGroup(state.dynamicObjects);
        // InstancedMesh supplies colors through instanceColor; BoxGeometry has no per-vertex color attribute.
        const material = new THREE.MeshStandardMaterial({ vertexColors: false, roughness: 0.32, metalness: 0.06 });
        state.dynamicMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), material, frame.positions.length);
        state.dynamicMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        state.dynamicObjects.add(state.dynamicMesh);
        state.dynamicMode = "boxes";
      }
      const matrix = new THREE.Matrix4();
      const scale = new THREE.Vector3();
      frame.positions.forEach((position, index) => {
        const size = getHalfSize(render, index, config);
        scale.set(size.x * 2, size.y * 2, size.z * 2);
        matrix.compose(new THREE.Vector3(position[0], position[1], position[2]), toQuaternion(frame.quaternions[index]), scale);
        state.dynamicMesh!.setMatrixAt(index, matrix);
        const materialIndex = render?.material_indices?.[index] ?? index;
        state.dynamicMesh!.setColorAt(index, render?.colors?.[materialIndex] ? new THREE.Color(render.colors[materialIndex]) : colorForIndex(materialIndex, frame.positions.length));
      });
      state.dynamicMesh.instanceMatrix.needsUpdate = true;
      if (state.dynamicMesh.instanceColor) state.dynamicMesh.instanceColor.needsUpdate = true;
    }
    fitCamera(state, frame, metadata, config);
    const center = frameCenter(frame);
    if (followCamera && center) {
      if (state.followTarget) {
        const delta = center.clone().sub(state.followTarget);
        state.camera.position.add(delta);
        state.controls.target.add(delta);
        state.controls.update();
      }
      state.followTarget = center;
    } else if (!followCamera) {
      state.followTarget = undefined;
    }
    clearGroup(state.overlays);
    if (showContacts && frame.contacts.length > 0) {
      const contactPositions = new Float32Array(frame.contacts.flatMap((point) => [point[0], point[1], point[2]]));
      const contactGeometry = new THREE.BufferGeometry();
      contactGeometry.setAttribute("position", new THREE.BufferAttribute(contactPositions, 3));
      state.overlays.add(new THREE.Points(contactGeometry, new THREE.PointsMaterial({ color: "#ffffff", size: 0.009, sizeAttenuation: true })));
    }
    if (showTrajectory && frame.positions.length > 0) {
      const center = frameCenter(frame);
      if (!center) return;
      if (state.previousTime >= 0 && frame.time < state.previousTime) state.trajectory = [];
      state.trajectory.push(center);
      state.previousTime = frame.time;
      if (state.trajectory.length > 1) state.overlays.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(state.trajectory), new THREE.LineBasicMaterial({ color: "#fff1a8", transparent: true, opacity: 0.8 })));
    }
  }, [config, frame, metadata, geometryMode, showContacts, showTrajectory, followCamera, ready]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state || !ready || cameraView === "orbit") return;
    const target = state.controls.target.clone();
    const distance = state.camera.position.distanceTo(target);
    if (cameraView === "front") state.camera.position.copy(target).add(new THREE.Vector3(0, -1, 0.1).normalize().multiplyScalar(distance));
    if (cameraView === "side") state.camera.position.copy(target).add(new THREE.Vector3(1, 0, 0.1).normalize().multiplyScalar(distance));
    if (cameraView === "top") state.camera.position.copy(target).add(new THREE.Vector3(0, 0, 1).multiplyScalar(distance));
    state.controls.update();
  }, [cameraView, ready]);

  const timeLabel = `${frame.time.toFixed(3)} s`;
  const physicalVersion = metadata?.engine_version || metadata?.mujoco_version;
  const physical = Boolean(physicalVersion && physicalVersion !== "none");
  return (
    <section className="viewport-card" data-testid="scene-view">
      <div className="viewport-toolbar"><div className="viewport-title"><span className="live-dot" />三维动力学视图 <span className="unit-chip">Z ↑</span></div><div className="viewport-actions"><label className="mode-select"><span>相机</span><select value={cameraView} onChange={(event) => onCameraViewChange(event.target.value as Props["cameraView"])}><option value="orbit">轨道</option><option value="front">正面</option><option value="side">侧面</option><option value="top">俯视</option></select></label></div></div>
      <div ref={mountRef} className="scene-canvas" />
      <div className="viewport-overlay"><span className="preview-badge">{geometryMode === "smooth" ? physical ? "彩虹带 · 插值表面 · MUJOCO 物理帧" : "彩虹带 · 插值表面 · 几何预览（未运行物理仿真）" : physical ? "碰撞几何 · MUJOCO 物理帧" : "碰撞几何 · 几何预览（未运行物理仿真）"}</span><span className="time-readout">{timeLabel}</span></div>
      <div className="axis-legend"><span><i className="axis-x" />X</span><span><i className="axis-y" />Y</span><span><i className="axis-z" />Z</span></div>
    </section>
  );
}
