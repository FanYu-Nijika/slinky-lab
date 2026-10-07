import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import type { Frame, MetricKey, PositionMetricKey, RunResult } from "../types";
import { METRIC_KEYS, METRIC_LABELS } from "../types";

interface Props {
  frames: Frame[];
  run?: RunResult | null;
  onUploadReference: (file: File, metric: MetricKey) => Promise<void>;
  onExportPng: (dataUrl: string) => void;
}

const energyKeys: MetricKey[] = ["kinetic_energy", "gravitational_energy", "cable_work", "damping_work", "contact_work"];
const materialPositionKeys: PositionMetricKey[] = ["top_z", "bottom_z", "com_z"];
const spatialPositionKeys: PositionMetricKey[] = ["spatial_top_z", "spatial_bottom_z", "com_z"];

export function ChartsPanel({ frames, run, onUploadReference, onExportPng }: Props) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.EChartsType>();
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>("com_z");
  const [chartMode, setChartMode] = useState<"position" | "energy">("position");
  const [positionMode, setPositionMode] = useState<"material" | "spatial">("material");
  const [referenceMetric, setReferenceMetric] = useState<MetricKey>("com_z");
  const [referenceSeries, setReferenceSeries] = useState<Array<[number, number]>>([]);
  const [referenceError, setReferenceError] = useState("");

  useEffect(() => {
    if (!chartRef.current) return undefined;
    chartInstance.current = echarts.init(chartRef.current, undefined, { renderer: "canvas" });
    const resize = () => chartInstance.current?.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chartInstance.current?.dispose();
      chartInstance.current = undefined;
    };
  }, []);

  useEffect(() => {
    const chart = chartInstance.current;
    if (!chart) return;
    const requestedKeys: Array<MetricKey | PositionMetricKey> = chartMode === "position" ? positionMode === "material" ? materialPositionKeys : spatialPositionKeys : energyKeys;
    const keys = requestedKeys.filter((key) => frames.some((frame) => Number.isFinite(frame.metrics[key]))) as Array<MetricKey | PositionMetricKey>;
    const colors = ["#54d5ff", "#ff9f7d", "#c3a6ff", "#7ce3b2", "#ffd166"];
    chart.setOption({
      animation: false,
      backgroundColor: "transparent",
      grid: { left: 42, right: 18, top: 20, bottom: 34 },
      tooltip: { trigger: "axis", backgroundColor: "#121e31", borderColor: "#334763", textStyle: { color: "#e6efff" } },
      legend: { top: 0, right: 2, textStyle: { color: "#93a5bf", fontSize: 10 }, itemWidth: 14, itemHeight: 7 },
      xAxis: { type: "value", name: "t / s", nameTextStyle: { color: "#71859f" }, axisLabel: { color: "#71859f", fontSize: 10 }, splitLine: { lineStyle: { color: "#1d2d42" } }, axisLine: { lineStyle: { color: "#334763" } } },
      yAxis: { type: "value", name: chartMode === "position" ? "高度 / m" : "能量 / J", nameTextStyle: { color: "#71859f" }, axisLabel: { color: "#71859f", fontSize: 10 }, splitLine: { lineStyle: { color: "#1d2d42" } }, axisLine: { lineStyle: { color: "#334763" } } },
      series: [
        ...keys.map((key, index) => ({ name: METRIC_LABELS[key], type: "line", showSymbol: false, smooth: false, lineStyle: { width: key === selectedMetric ? 2.6 : 1.3, color: colors[index] }, itemStyle: { color: colors[index] }, data: frames.map((frame) => [frame.time, frame.metrics[key] ?? null]) })),
        ...(referenceSeries.length && keys.some((key) => key === referenceMetric) ? [{ name: "实验参考", type: "line", showSymbol: false, smooth: false, lineStyle: { width: 1.6, type: "dashed", color: "#f4c66e" }, itemStyle: { color: "#f4c66e" }, data: referenceSeries }] : []),
      ],
    }, true);
  }, [frames, chartMode, positionMode, selectedMetric, referenceMetric, referenceSeries]);

  const exportPng = () => {
    const chart = chartInstance.current;
    if (chart) onExportPng(chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#0b1422" }));
  };

  const current = frames.length ? frames[frames.length - 1].metrics : {};
  const hasSpatialSeries = frames.some((frame) => Number.isFinite(frame.metrics.spatial_top_z) || Number.isFinite(frame.metrics.spatial_bottom_z));
  const localRmse = computeRmse(frames, referenceSeries, referenceMetric);
  return (
    <section className="analysis-card" data-testid="charts-panel">
      <div className="analysis-header">
        <div>
          <span className="eyebrow">时间序列分析</span>
          <h2>运动与能量</h2>
        </div>
        <div className="analysis-actions">
          <div className="segmented compact"><button className={chartMode === "position" ? "active" : ""} onClick={() => setChartMode("position")}>关键位置</button><button className={chartMode === "energy" ? "active" : ""} onClick={() => setChartMode("energy")}>能量/做功</button></div>
          {chartMode === "position" && <div className="segmented compact"><button className={positionMode === "material" ? "active" : ""} onClick={() => setPositionMode("material")}>材料端点</button><button disabled={!hasSpatialSeries} className={positionMode === "spatial" ? "active" : ""} onClick={() => setPositionMode("spatial")}>空间极值</button></div>}
          <button className="icon-button" title="导出 PNG" onClick={exportPng}>↗ PNG</button>
        </div>
      </div>
      <div ref={chartRef} className="chart-canvas" />
      <div className="metric-strip">
        <div><span>材料上端</span><strong>{formatMetric(current.top_z)} m</strong></div>
        <div><span>材料下端</span><strong>{formatMetric(current.bottom_z)} m</strong></div>
        <div><span>空间最高</span><strong>{formatMetric(current.spatial_top_z)} m</strong></div>
        <div><span>空间最低</span><strong>{formatMetric(current.spatial_bottom_z)} m</strong></div>
        <div><span>质心</span><strong>{formatMetric(current.com_z)} m</strong></div>
      </div>
      <div className="reference-row">
        <span>实验数据对照</span>
        <select className="select-input tiny" value={referenceMetric} onChange={(event) => setReferenceMetric(event.target.value as MetricKey)}>
          {METRIC_KEYS.map((key) => <option key={key} value={key}>{METRIC_LABELS[key]}</option>)}
        </select>
        <label className="upload-button">上传 CSV<input type="file" accept=".csv,text/csv" onChange={(event) => { const file = event.target.files?.[0]; if (file) void handleReferenceFile(file, referenceMetric, onUploadReference, setReferenceSeries, setReferenceError, setSelectedMetric, setChartMode); event.currentTarget.value = ""; }} /></label>
        {referenceError && <span className="inline-error compact-error">{referenceError}</span>}
        {(localRmse !== undefined || run?.summary.reference_rmse !== undefined) && <span className="rmse-chip">RMSE {(localRmse ?? run?.summary.reference_rmse ?? 0).toPrecision(4)} · 共同时间段</span>}
      </div>
    </section>
  );
}

async function handleReferenceFile(file: File, metric: MetricKey, upload: (file: File, metric: MetricKey) => Promise<void>, setSeries: (series: Array<[number, number]>) => void, setError: (error: string) => void, setMetric: (metric: MetricKey) => void, setMode: (mode: "position" | "energy") => void) {
  try {
    const series = parseReferenceCsv(await file.text(), metric);
    setSeries(series);
    setMetric(metric);
    setMode(energyKeys.includes(metric) ? "energy" : "position");
    setError("");
    await upload(file, metric);
  } catch (cause) {
    setError(cause instanceof Error ? cause.message : "CSV 读取失败");
  }
}

function parseReferenceCsv(text: string, metric: MetricKey): Array<[number, number]> {
  const rows = text.split(/\r?\n/).map((line) => line.trim()).filter((line) => line && !line.startsWith("#")).map((line) => line.split(/[,;\t]/).map((cell) => cell.trim()));
  if (rows.length < 2) throw new Error("CSV 至少需要一行表头和一行数据");
  const header = rows[0].map((cell) => cell.toLowerCase());
  const timeIndex = header.findIndex((cell) => ["time", "t", "timestamp", "时间"].includes(cell));
  const valueIndex = header.findIndex((cell) => cell === metric || cell === METRIC_LABELS[metric].toLowerCase());
  const tIndex = timeIndex >= 0 ? timeIndex : 0;
  const vIndex = valueIndex >= 0 ? valueIndex : tIndex === 0 ? 1 : 0;
  const series = rows.slice(1).map((row) => [Number(row[tIndex]), Number(row[vIndex])] as [number, number]).filter(([time, value]) => Number.isFinite(time) && Number.isFinite(value));
  if (series.length < 2) throw new Error("CSV 中没有找到有效的时间/指标数值");
  return series.sort((a, b) => a[0] - b[0]);
}

function computeRmse(frames: Frame[], reference: Array<[number, number]>, metric: MetricKey): number | undefined {
  if (frames.length < 2 || reference.length < 2) return undefined;
  const minTime = Math.max(frames[0].time, reference[0][0]);
  const maxTime = Math.min(frames[frames.length - 1].time, reference[reference.length - 1][0]);
  if (!(maxTime > minTime)) return undefined;
  const errors: number[] = [];
  for (const frame of frames) {
    if (frame.time < minTime || frame.time > maxTime) continue;
    const value = frame.metrics[metric];
    if (!Number.isFinite(value)) continue;
    const refValue = interpolate(reference, frame.time);
    if (refValue !== undefined) errors.push(value - refValue);
  }
  return errors.length ? Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / errors.length) : undefined;
}

function interpolate(series: Array<[number, number]>, time: number): number | undefined {
  let right = series.findIndex(([sampleTime]) => sampleTime >= time);
  if (right < 0) return undefined;
  if (right === 0) return series[0][1];
  const left = series[right - 1];
  const next = series[right];
  const span = next[0] - left[0];
  return span > 0 ? left[1] + ((time - left[0]) / span) * (next[1] - left[1]) : next[1];
}

function formatMetric(value: number | undefined, digits = 3): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}
