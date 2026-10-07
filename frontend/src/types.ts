export type Scenario = "drop" | "stairs";
export type Profile = "preview" | "fine";
export type RunStatus =
  | "queued"
  | "preparing"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

export interface Material {
  turns: number;
  radius: number;
  strip_width: number;
  strip_thickness: number;
  pitch: number;
  mass: number;
  young_modulus: number;
  shear_modulus: number;
  damping: number;
  friction: number;
  self_friction?: number;
}

export interface Scene {
  gravity: number;
  step_height: number;
  step_depth: number;
  step_width: number;
  step_count: number;
  tilt_deg: number;
  initial_angular_velocity: number;
  initial_forward_velocity: number;
  initial_lateral_velocity?: number;
  launch_offset: number;
  settle_time: number;
}

export interface Numerics {
  profile: Profile;
  duration: number;
  sample_hz: number;
  segments_per_turn: number | null;
  timestep: number | null;
  max_wall_seconds: number;
  contact_time_constant?: number;
}

export interface RunConfig {
  schema_version: 1;
  name: string;
  scenario: Scenario;
  material: Material;
  scene: Scene;
  numerics: Numerics;
  provenance: Record<string, string>;
}

export interface Frame {
  time: number;
  positions: number[][];
  quaternions: number[][];
  contacts: number[][];
  metrics: Record<string, number>;
  contact_details?: { position: number[]; geom_a: number; geom_b: number; normal_force: number;
    kind: string; stair_step: number | null; material_index: number | null; surface: string }[];
}

export interface GeomMetadata {
  name?: string;
  type?: "box" | "plane" | "sphere" | string;
  half_size?: number[];
  half_sizes?: number[];
  center?: number[];
  position?: number[];
  quaternion?: number[];
  static?: boolean;
  material_index?: number;
  color?: string;
}

export interface RenderMetadata {
  half_sizes?: number[][];
  types?: string[];
  names?: string[];
  static?: boolean[];
  positions?: number[][];
  quaternions?: number[][];
  material_indices?: number[];
  colors?: string[];
  geoms?: GeomMetadata[];
  static_geoms?: GeomMetadata[];
  render?: RenderMetadata;
  engine_version?: string;
  mujoco_version?: string;
  model_version?: string;
  [key: string]: unknown;
}

export interface RunSummary {
  frames?: number;
  duration?: number;
  top_z?: number;
  bottom_z?: number;
  spatial_top_z?: number;
  spatial_bottom_z?: number;
  material_top_z?: number;
  material_bottom_z?: number;
  material_first_z?: number;
  material_last_z?: number;
  com_z?: number;
  steps_descended?: number;
  max_penetration?: number;
  kinetic_energy?: number;
  gravitational_energy?: number;
  cable_work?: number;
  damping_work?: number;
  contact_work?: number;
  prepare_status?: string;
  settle_force_residual?: number | null;
  settle_anchor_error_m?: number | null;
  bottom_onset_time?: number | null;
  collapse_time?: number | null;
  movement_classification?: string;
  step_events?: StepEvent[];
  support_diagnostics?: SupportDiagnostics;
  numerical_validation?: "passed" | "failed" | "pending" | string;
  experimental_validation?: "supported" | "unsupported" | "pending" | string;
  reference_rmse?: number;
  [key: string]: unknown;
}

export interface StepEvent {
  time?: number;
  step?: number;
  end?: string;
  [key: string]: unknown;
}

export interface EndpointSupport {
  label?: string;
  time?: number;
  [key: string]: unknown;
}

export interface SupportEvent {
  step?: number;
  first_touch_time?: number | null;
  first_touch_label?: string | null;
  first_sustained_time?: number | null;
  first_sustained_label?: string | null;
  endpoint_supports?: EndpointSupport[];
  ambiguous?: boolean;
  ambiguity_reasons?: string[];
  [key: string]: unknown;
}

export interface SupportDiagnostics {
  candidate_flip_count?: number;
  confirmed_flip_count?: number;
  events?: SupportEvent[];
  [key: string]: unknown;
}

export interface RunResult {
  run_id: string;
  status: RunStatus | string;
  config: RunConfig;
  summary: RunSummary;
  artifacts: string[];
  error?: string | null;
  progress?: number;
  metadata?: RenderMetadata;
  frames?: Frame[];
}

export interface PresetInfo {
  id?: string;
  name?: string;
  label?: string;
  config?: RunConfig;
  description?: string;
}

export interface SweepAxis {
  parameter: string;
  values: number[];
}

export interface SweepConfig {
  name: string;
  base: RunConfig;
  axes: SweepAxis[];
}

export interface SweepRun {
  parameters: Record<string, number>;
  run_id?: string;
  status: RunStatus | string;
  summary?: RunSummary;
}

export interface SweepResult {
  sweep_id: string;
  status: RunStatus | string;
  runs: SweepRun[];
  summary?: RunSummary & { results?: SweepRun[] };
  progress?: number;
  error?: string | null;
}

export const DEFAULT_MATERIAL: Material = {
  turns: 12,
  radius: 0.03,
  strip_width: 0.003,
  strip_thickness: 0.0015,
  pitch: 0.0017,
  mass: 0.0487,
  young_modulus: 1e8,
  shear_modulus: 3.7e7,
  damping: 0.00001,
  friction: 0.5,
  self_friction: 0.5,
};

export const DEFAULT_SCENE: Scene = {
  gravity: 9.81,
  step_height: 0.04,
  step_depth: 0.08,
  step_width: 0.3,
  step_count: 6,
  tilt_deg: 25,
  initial_angular_velocity: 0,
  initial_forward_velocity: 0,
  initial_lateral_velocity: 0,
  launch_offset: 0,
  settle_time: 2,
};

export const DEFAULT_NUMERICS: Numerics = {
  profile: "preview",
  duration: 1.5,
  sample_hz: 60,
  segments_per_turn: null,
  timestep: null,
  max_wall_seconds: 600,
  contact_time_constant: 0.003,
};

export const DEFAULT_CONFIG: RunConfig = {
  schema_version: 1,
  name: "彩虹圈实验",
  scenario: "drop",
  material: { ...DEFAULT_MATERIAL },
  scene: { ...DEFAULT_SCENE },
  numerics: { ...DEFAULT_NUMERICS },
  provenance: { preset: "illustrative: not experimentally calibrated" },
};

// Keep the verified short drop case available when the API is unreachable. The
// service remains authoritative when it is online, but this gives an offline
// first view the same stable 3-turn mesh and 12.5 µs step used for validation.
export const DEFAULT_DROP_VALIDATION_CONFIG: RunConfig = {
  schema_version: 1,
  name: "下落数值验证小算例",
  scenario: "drop",
  material: {
    ...DEFAULT_MATERIAL,
    turns: 3,
    mass: 0.012,
    pitch: 0.002,
    young_modulus: 1e8,
    shear_modulus: 1e8 / 2.7,
  },
  scene: { ...DEFAULT_SCENE, tilt_deg: 0 },
  numerics: {
    ...DEFAULT_NUMERICS,
    duration: 0.18,
    sample_hz: 100,
    segments_per_turn: 16,
    timestep: 0.0000125,
  },
  provenance: {
    preset: "drop-validation",
    geometry_material: "explicit demonstration assumptions, not measured plastic slinky",
    experimental_support: "not established",
  },
};

export const METRIC_KEYS = [
  "top_z",
  "bottom_z",
  "com_z",
  "kinetic_energy",
  "gravitational_energy",
  "cable_work",
  "damping_work",
  "contact_work",
] as const;

export type MetricKey = (typeof METRIC_KEYS)[number];

export const POSITION_METRIC_KEYS = ["top_z", "bottom_z", "spatial_top_z", "spatial_bottom_z", "com_z"] as const;

export type PositionMetricKey = (typeof POSITION_METRIC_KEYS)[number];

export const METRIC_LABELS: Record<MetricKey | PositionMetricKey, string> = {
  top_z: "材料上端高度",
  bottom_z: "材料下端高度",
  spatial_top_z: "空间最高点",
  spatial_bottom_z: "空间最低点",
  com_z: "质心高度",
  kinetic_energy: "动能",
  gravitational_energy: "重力势能",
  cable_work: "弹性做功",
  damping_work: "阻尼做功",
  contact_work: "接触做功",
};
