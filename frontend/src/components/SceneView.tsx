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
  const fallback = [config.material.strip_width / 2, config.material.strip_thickness / 2, 0.0018];
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

function smoothRibbon(frame: Frame, config: RunConfig): THREE.Mesh | null {
  if (frame.positions.length < 2) return null;
  const points = frame.positions.map((point) => new THREE.Vector3(point[0], point[1], point[2]));
  const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
  const tubularSegments = Math.min(180, Math.max(24, points.length * 2));
  const geometry = new THREE.TubeGeometry(curve, tubularSegments, Math.max(config.material.strip_width * 0.54, 0.0012), 8, false);
  const position = geometry.getAttribute("position");
  const colors = new Float32Array(position.count * 3);
  for (let vertex = 0; vertex < position.count; vertex += 1) {
    const band = Math.min(points.length - 1, Math.floor((vertex / Math.max(position.count - 1, 1)) * points.length));
    const color = colorForIndex(band, points.length);
    colors[vertex * 3] = color.r;
    colors[vertex * 3 + 1] = color.g;
    colors[vertex * 3 + 2] = color.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.28, metalness: 0.08 }));
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
  state.camera.position.copy(center.clone().add(new THREE.Vector3(0.9, -1.1, 0.75).normalize().multiplyScalar(radius * 2.6)));
  state.camera.near = Math.max(radius / 100, 0.0001);
  state.camera.far = Math.max(radius * 20, 2);
  state.camera.updateProjectionMatrix();
  state.controls.update();
  state.hasFit = true;
}

export function SceneView({ config, frame, metadata, runKey, showContacts, showTrajectory, geometryMode, cameraView, onCameraViewChange }: Props) {
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
    const ambient = new THREE.HemisphereLight("#dbeafe", "#0b1020", 1.8);
    scene.add(ambient);
    const key = new THREE.DirectionalLight("#ffffff", 2.8);
    key.position.set(2, -3, 4);
    scene.add(key);
    const rim = new THREE.PointLight("#5dd7ff", 1.2, 4);
    rim.position.set(-1.5, 0.5, 1.2);
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
    state.hasFit = false;
  }, [runKey, ready]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state || !ready) return;
    clearGroup(state.staticObjects);
    renderStaticGeoms(state.staticObjects, metadata, config);
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
      const ribbon = smoothRibbon(frame, config);
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
    clearGroup(state.overlays);
    if (showContacts && frame.contacts.length > 0) {
      const contactPositions = new Float32Array(frame.contacts.flatMap((point) => [point[0], point[1], point[2]]));
      const contactGeometry = new THREE.BufferGeometry();
      contactGeometry.setAttribute("position", new THREE.BufferAttribute(contactPositions, 3));
      state.overlays.add(new THREE.Points(contactGeometry, new THREE.PointsMaterial({ color: "#ffffff", size: 0.009, sizeAttenuation: true })));
    }
    if (showTrajectory && frame.positions.length > 0) {
      const center = frame.positions.reduce((acc, point) => acc.add(new THREE.Vector3(point[0], point[1], point[2])), new THREE.Vector3()).multiplyScalar(1 / frame.positions.length);
      if (state.previousTime >= 0 && frame.time < state.previousTime) state.trajectory = [];
      state.trajectory.push(center);
      state.previousTime = frame.time;
      if (state.trajectory.length > 1) state.overlays.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(state.trajectory), new THREE.LineBasicMaterial({ color: "#fff1a8", transparent: true, opacity: 0.8 })));
    }
  }, [config, frame, metadata, geometryMode, showContacts, showTrajectory, ready]);

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
      <div className="viewport-overlay"><span className="preview-badge">{physical ? "MUJOCO · 物理帧" : geometryMode === "smooth" ? "平滑形状示意 · 未运行物理仿真" : "几何预览 · 未运行物理仿真"}</span><span className="time-readout">{timeLabel}</span></div>
      <div className="axis-legend"><span><i className="axis-x" />X</span><span><i className="axis-y" />Y</span><span><i className="axis-z" />Z</span></div>
    </section>
  );
}
