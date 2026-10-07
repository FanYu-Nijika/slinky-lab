import { useState } from "react";
import type { InitialPose, Material, Numerics, RunConfig, Scene } from "../types";

interface Props {
  config: RunConfig;
  presets: Array<{ id?: string; name?: string; label?: string; config?: RunConfig; description?: string }>;
  onChange: (config: RunConfig) => void;
  onPreset: (name: string) => void;
  collapsed: boolean;
}

type NumberFieldProps = {
  label: string;
  value: number;
  step?: number;
  min?: number;
  max?: number;
  unit?: string;
  onChange: (value: number) => void;
};

function NumberField({ label, value, step = 0.001, min, max, unit, onChange }: NumberFieldProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="field-input">
        <input
          type="number"
          value={Number.isFinite(value) ? value : ""}
          step={step}
          min={min}
          max={max}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        {unit && <small>{unit}</small>}
      </div>
    </label>
  );
}

function Section({ title, children, defaultOpen = true }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="param-section" open={defaultOpen}>
      <summary>
        <span>{title}</span>
        <span className="summary-mark">⌄</span>
      </summary>
      <div className="param-section-body">{children}</div>
    </details>
  );
}

export function ParameterPanel({ config, presets, onChange, onPreset, collapsed }: Props) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const updateMaterial = (patch: Partial<Material>) => onChange({ ...config, material: { ...config.material, ...patch } });
  const updateScene = (patch: Partial<Scene>) => onChange({ ...config, scene: { ...config.scene, ...patch } });
  const updateNumerics = (patch: Partial<Numerics>) => onChange({ ...config, numerics: { ...config.numerics, ...patch } });
  const initialPose: InitialPose = config.scene.initial_pose ?? "tilted";
  const archEndTurns = config.scene.arch_end_turns ?? 2;
  const automaticArchRise = config.scene.step_depth / 2;
  const chooseProfile = (profile: Numerics["profile"]) => {
    if (profile === config.numerics.profile) return;
    const segments = config.numerics.segments_per_turn ?? (config.numerics.profile === "fine" ? 24 : 12);
    const dt = config.numerics.timestep ?? (config.numerics.profile === "fine" ? 0.00005 : 0.0002);
    // Keep preset-specific refinement relative to its current mesh. Bending
    // frequencies rise as segment length shrinks, so halve h and quarter dt.
    const refine = profile === "fine";
    updateNumerics({ profile, segments_per_turn: refine ? Math.min(64, segments * 2) : Math.max(8, Math.floor(segments / 2)),
                     timestep: refine ? Math.max(0.000001, dt / 4) : Math.min(0.002, dt * 4) });
  };

  return (
    <aside className={`panel left-panel ${collapsed ? "is-collapsed" : ""}`} data-testid="parameter-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">实验设置</span>
          <h2>参数</h2>
        </div>
        <span className="panel-count">SI</span>
      </div>
      <div className="panel-scroll">
        <Section title="实验预设">
          <div className="preset-grid">
            <button className={`preset-card ${config.scenario === "drop" ? "selected" : ""}`} onClick={() => onPreset("drop")}>
              <span className="preset-icon">↓</span>
              <span>悬挂下落</span>
              <small>释放顶部约束</small>
            </button>
            <button className={`preset-card ${config.scenario === "stairs" ? "selected" : ""}`} onClick={() => onPreset("stairs")}>
              <span className="preset-icon">⌁</span>
              <span>翻转下楼梯</span>
              <small>带接触的运动</small>
            </button>
          </div>
          {presets.length > 0 && (
            <select className="select-input preset-select" aria-label="后端预设" value="" onChange={(event) => onPreset(event.target.value)}>
              <option value="">选择后端预设…</option>
              {presets.map((preset) => (
                <option key={preset.id || preset.name} value={preset.id || preset.name}>{preset.label || preset.name}</option>
              ))}
            </select>
          )}
        </Section>

        <Section title="场景">
          <label className="field">
            <span>名称</span>
            <input className="text-input" value={config.name} onChange={(event) => onChange({ ...config, name: event.target.value })} />
          </label>
          <NumberField label="重力" value={config.scene.gravity} min={0} max={30} step={0.01} unit="m/s²" onChange={(value) => updateScene({ gravity: value })} />
          {config.scenario === "stairs" && (
            <>
              <NumberField label="阶高" value={config.scene.step_height} min={0.002} max={0.3} step={0.001} unit="m" onChange={(value) => updateScene({ step_height: value })} />
              <NumberField label="阶深" value={config.scene.step_depth} min={0.015} max={0.5} step={0.001} unit="m" onChange={(value) => updateScene({ step_depth: value })} />
              <NumberField label="阶宽" value={config.scene.step_width} min={0.05} max={2} step={0.01} unit="m" onChange={(value) => updateScene({ step_width: value })} />
              <NumberField label="阶梯数量" value={config.scene.step_count} min={3} max={20} step={1} unit="阶" onChange={(value) => updateScene({ step_count: value })} />
              <label className="field">
                <span>起步姿态</span>
                <select className="select-input" aria-label="起步姿态" value={initialPose} onChange={(event) => updateScene({ initial_pose: event.target.value as InitialPose })}>
                  <option value="tilted">整体倾斜</option>
                  <option value="arched">手动拱形释放</option>
                </select>
              </label>
              {initialPose === "tilted" ? (
                <NumberField label="初始倾角" value={config.scene.tilt_deg} min={-80} max={80} step={1} unit="°" onChange={(value) => updateScene({ tilt_deg: value })} />
              ) : (
                <>
                  <label className="field">
                    <span>拱高来源</span>
                    <select className="select-input" aria-label="拱高来源" value={config.scene.arch_rise == null ? "auto" : "custom"} onChange={(event) => updateScene({ arch_rise: event.target.value === "auto" ? null : automaticArchRise })}>
                      <option value="auto">自动（阶深一半）</option>
                      <option value="custom">自定义拱高</option>
                    </select>
                  </label>
                  {config.scene.arch_rise == null ? (
                    <div className="field"><span>拱高</span><div className="field-input"><span>{automaticArchRise.toFixed(4)} m（自动）</span></div></div>
                  ) : (
                    <NumberField label="拱高" value={config.scene.arch_rise} min={0} max={1} step={0.001} unit="m" onChange={(value) => updateScene({ arch_rise: value })} />
                  )}
                  <NumberField label="两端压缩圈数" value={archEndTurns} min={1} max={Math.max(1, Math.floor((config.material.turns - 2) / 2))} step={1} unit="圈" onChange={(value) => updateScene({ arch_end_turns: value })} />
                  <NumberField label="自由端间隙" value={config.scene.arch_free_clearance ?? 0.025} min={0} max={0.5} step={0.001} unit="m" onChange={(value) => updateScene({ arch_free_clearance: value })} />
                  <div className="source-note" style={{ padding: "8px 0 0" }}><span className="source-dot" /><div>初始弯曲储存弹性能，放手后由动力学产生运动。拱形释放不使用初始倾角；初始速度仍按数值输入。</div></div>
                  {config.material.turns < 2 * archEndTurns + 2 && <div className="source-note" style={{ padding: "4px 0 0" }}><span className="source-dot" /><div>当前圈数不足：至少需要 {2 * archEndTurns + 2} 圈才能容纳两端压缩圈。</div></div>}
                </>
              )}
              <NumberField label="初始前向速度" value={config.scene.initial_forward_velocity} min={-2} max={2} step={0.01} unit="m/s" onChange={(value) => updateScene({ initial_forward_velocity: value })} />
              <NumberField label="初始侧向速度" value={config.scene.initial_lateral_velocity ?? 0} min={-2} max={2} step={0.01} unit="m/s" onChange={(value) => updateScene({ initial_lateral_velocity: value })} />
              <NumberField label="初始角速度" value={config.scene.initial_angular_velocity} min={-20} max={20} step={0.1} unit="rad/s" onChange={(value) => updateScene({ initial_angular_velocity: value })} />
              <NumberField label="边缘摆放偏移" value={config.scene.launch_offset} min={-0.1} max={0.2} step={0.001} unit="m" onChange={(value) => updateScene({ launch_offset: value })} />
            </>
          )}
        </Section>

        <Section title="彩虹圈材料">
          <div className="field-grid">
            <NumberField label="圈数" value={config.material.turns} min={3} max={80} step={1} unit="圈" onChange={(value) => updateMaterial({ turns: value })} />
            <NumberField label="半径" value={config.material.radius} min={0.005} max={0.15} step={0.001} unit="m" onChange={(value) => updateMaterial({ radius: value })} />
          </div>
          <div className="field-grid">
            <NumberField label="带宽" value={config.material.strip_width} min={0.0002} max={0.02} step={0.0001} unit="m" onChange={(value) => updateMaterial({ strip_width: value })} />
            <NumberField label="带厚" value={config.material.strip_thickness} min={0.0002} max={0.008} step={0.0001} unit="m" onChange={(value) => updateMaterial({ strip_thickness: value })} />
          </div>
          <NumberField label="螺距" value={config.material.pitch} min={0.0002} max={0.02} step={0.0001} unit="m" onChange={(value) => updateMaterial({ pitch: value })} />
          <NumberField label="总质量" value={config.material.mass} min={0.001} max={2} step={0.001} unit="kg" onChange={(value) => updateMaterial({ mass: value })} />
        </Section>

        <Section title="弹性与接触" defaultOpen={false}>
          <NumberField label="杨氏模量 E" value={config.material.young_modulus} min={1e4} max={3e11} step={1e6} unit="Pa" onChange={(value) => updateMaterial({ young_modulus: value })} />
          <NumberField label="剪切模量 G" value={config.material.shear_modulus} min={1e3} max={1.5e11} step={1e6} unit="Pa" onChange={(value) => updateMaterial({ shear_modulus: value })} />
          <NumberField label="铰接阻尼" value={config.material.damping} min={0} max={0.1} step={0.000001} unit="Nm·s" onChange={(value) => updateMaterial({ damping: value })} />
          <NumberField label="台阶/地面摩擦" value={config.material.friction} min={0} max={3} step={0.05} onChange={(value) => updateMaterial({ friction: value })} />
          <NumberField label="圈间摩擦" value={config.material.self_friction ?? 0.5} min={0} max={3} step={0.05} onChange={(value) => updateMaterial({ self_friction: value })} />
          {config.scenario === "drop" && <NumberField label="平衡松弛时长" value={config.scene.settle_time} min={0.05} max={15} step={0.1} unit="s" onChange={(value) => updateScene({ settle_time: value })} />}
        </Section>

        <Section title="计算精度">
          <div className="segmented">
            <button className={config.numerics.profile === "preview" ? "active" : ""} onClick={() => chooseProfile("preview")}>快速预览</button>
            <button className={config.numerics.profile === "fine" ? "active" : ""} onClick={() => chooseProfile("fine")}>精细研究</button>
          </div>
          <NumberField label="仿真时长" value={config.numerics.duration} min={0.02} max={30} step={0.1} unit="s" onChange={(value) => updateNumerics({ duration: value })} />
          <NumberField label="采样频率" value={config.numerics.sample_hz} min={10} max={120} step={1} unit="Hz" onChange={(value) => updateNumerics({ sample_hz: value })} />
          <button className="text-button" onClick={() => setShowAdvanced((open) => !open)}>{showAdvanced ? "收起高级数值设置" : "展开高级数值设置"}</button>
          {showAdvanced && (
            <>
              <NumberField label="每圈段数" value={config.numerics.segments_per_turn || (config.numerics.profile === "fine" ? 24 : 12)} min={8} max={64} step={1} unit="段" onChange={(value) => updateNumerics({ segments_per_turn: value })} />
              <NumberField label="积分步长" value={config.numerics.timestep || (config.numerics.profile === "fine" ? 0.00005 : 0.0002)} min={0.000001} max={0.002} step={0.000001} unit="s" onChange={(value) => updateNumerics({ timestep: value })} />
              <NumberField label="接触时间常数" value={config.numerics.contact_time_constant ?? 0.003} min={0.00005} max={0.03} step={0.0001} unit="s" onChange={(value) => updateNumerics({ contact_time_constant: value })} />
              <NumberField label="计算时间上限" value={config.numerics.max_wall_seconds} min={5} max={14400} step={60} unit="s" onChange={(value) => updateNumerics({ max_wall_seconds: value })} />
            </>
          )}
        </Section>

        <div className="source-note">
          <span className="source-dot" />
          <div><strong>参数来源</strong><br />{config.provenance.preset || "自定义参数"}</div>
        </div>
      </div>
    </aside>
  );
}
