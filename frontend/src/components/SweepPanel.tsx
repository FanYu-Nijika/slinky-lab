import { useState } from "react";
import type { RunConfig, SweepAxis, SweepResult } from "../types";

interface Props {
  base: RunConfig;
  sweep?: SweepResult | null;
  onStart: (config: { name: string; base: RunConfig; axes: SweepAxis[] }) => Promise<void>;
}

const PARAMETERS = [
  { value: "material.radius", label: "半径" },
  { value: "material.pitch", label: "螺距" },
  { value: "material.mass", label: "质量" },
  { value: "material.friction", label: "摩擦" },
  { value: "scene.step_height", label: "阶高" },
  { value: "scene.step_depth", label: "阶深" },
  { value: "scene.tilt_deg", label: "初始倾角" },
];

export function SweepPanel({ base, sweep, onStart }: Props) {
  const [open, setOpen] = useState(false);
  const [twoDimensional, setTwoDimensional] = useState(false);
  const [name, setName] = useState("参数扫描");
  const [axisOne, setAxisOne] = useState("material.radius");
  const [valuesOne, setValuesOne] = useState("0.024, 0.03, 0.036");
  const [axisTwo, setAxisTwo] = useState("material.friction");
  const [valuesTwo, setValuesTwo] = useState("0.2, 0.5, 0.8");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const start = async () => {
    let parsedOne: number[];
    let parsedTwo: number[] = [];
    try {
      parsedOne = parseValues(valuesOne);
      if (twoDimensional) parsedTwo = parseValues(valuesTwo);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "请输入有效的数字列表");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const axes: SweepAxis[] = [{ parameter: axisOne, values: parsedOne }];
      if (twoDimensional) axes.push({ parameter: axisTwo, values: parsedTwo });
      await onStart({ name, base, axes });
      setOpen(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "扫描创建失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className={`sweep-panel ${open ? "expanded" : ""}`} data-testid="sweep-panel">
      <button className="sweep-heading" onClick={() => setOpen((value) => !value)}><span><span className="sweep-icon">⌘</span><strong>参数扫描</strong><small>单参数或双参数网格</small></span><span>{open ? "⌃" : "⌄"}</span></button>
      {open && <div className="sweep-body">
        <label className="field"><span>扫描名称</span><input className="text-input" value={name} onChange={(event) => setName(event.target.value)} /></label>
        <div className="sweep-dimension"><span>扫描维度</span><div className="segmented compact"><button className={!twoDimensional ? "active" : ""} onClick={() => setTwoDimensional(false)}>1D</button><button className={twoDimensional ? "active" : ""} onClick={() => setTwoDimensional(true)}>2D</button></div></div>
        <AxisEditor label="X 轴" parameter={axisOne} values={valuesOne} onParameter={setAxisOne} onValues={setValuesOne} />
        {twoDimensional && <AxisEditor label="Y 轴" parameter={axisTwo} values={valuesTwo} onParameter={setAxisTwo} onValues={setValuesTwo} />}
        {error && <div className="inline-error">{error}</div>}
        <button className="primary-button full" disabled={busy} onClick={() => void start()}>{busy ? "提交中…" : "开始网格扫描"}</button>
      </div>}
      {sweep && <div className="sweep-result"><div><span>最近扫描</span><strong>{statusLabel(sweep.status)}</strong></div><div className="run-progress"><i style={{ width: `${Math.min(100, Math.max(0, (sweep.progress || completed(sweep)) * 100))}%` }} /></div><small>{sweep.runs?.length || 0} 组组合 · {sweep.sweep_id.slice(0, 8)}</small></div>}
    </section>
  );
}

function AxisEditor({ label, parameter, values, onParameter, onValues }: { label: string; parameter: string; values: string; onParameter: (value: string) => void; onValues: (value: string) => void }) {
  return <div className="axis-editor"><span>{label}</span><select className="select-input" value={parameter} onChange={(event) => onParameter(event.target.value)}>{PARAMETERS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select><input className="text-input" value={values} onChange={(event) => onValues(event.target.value)} placeholder="0.02, 0.03, 0.04" /></div>;
}

function parseValues(raw: string): number[] {
  const tokens = raw.split(/[,\s]+/).map((token) => token.trim()).filter(Boolean);
  if (!tokens.length) throw new Error("请输入有效的数字列表");
  if (tokens.length > 20) throw new Error("每个扫描轴最多 20 个取值");
  const values = tokens.map(Number);
  if (values.some((value) => !Number.isFinite(value))) throw new Error("扫描值必须全部是数字");
  return values;
}

function completed(sweep: SweepResult): number {
  const runs = sweep.runs || [];
  return runs.length ? runs.filter((run) => ["completed", "failed", "cancelled"].includes(run.status)).length / runs.length : 0;
}

function statusLabel(status: string): string {
  return ({ queued: "排队中", preparing: "准备中", running: "扫描中", completed: "已完成", failed: "失败" } as Record<string, string>)[status] || status;
}
