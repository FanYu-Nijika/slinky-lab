import type { RunResult, RunSummary, StepEvent, SupportEvent } from "../types";

interface Props {
  runs: RunResult[];
  selectedId?: string;
  selectedRun?: RunResult | null;
  onSelect: (run: RunResult) => void;
  onExport: (name: string) => void;
}

const statusLabels: Record<string, string> = {
  queued: "排队中",
  preparing: "准备模型",
  running: "计算中",
  paused: "已暂停",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "中断",
};

function scenarioLabel(run: RunResult): string {
  return run.config?.scenario === "stairs" ? "翻转下楼梯" : "悬挂下落";
}

export function RunHistory({ runs, selectedId, selectedRun, onSelect, onExport }: Props) {
  return (
    <aside className="panel right-panel" data-testid="run-history">
      <div className="panel-heading">
        <div><span className="eyebrow">实验记录</span><h2>运行历史</h2></div>
        <span className="panel-count">{runs.length.toString().padStart(2, "0")}</span>
      </div>
      <div className="panel-scroll history-scroll">
        {selectedRun && <ResearchSummary run={selectedRun} />}
        {runs.length === 0 && <div className="empty-history"><div className="empty-orbit">◌</div><strong>还没有运行记录</strong><span>从左侧参数开始一次研究运行</span></div>}
        {runs.map((run) => (
          <button key={run.run_id} className={`run-card ${selectedId === run.run_id ? "selected" : ""}`} onClick={() => onSelect(run)}>
            <div className="run-card-top"><span className={`status-dot status-${run.status}`} /><span className="run-name">{run.config?.name || run.run_id.slice(0, 8)}</span><span className={`status-label status-text-${run.status}`}>{statusLabels[run.status] || run.status}</span></div>
            <div className="run-card-meta"><span>{scenarioLabel(run)}</span><span>{formatDate(run.run_id)}</span></div>
            {run.progress !== undefined && run.status === "running" && <div className="run-progress"><i style={{ width: `${Math.min(100, Math.max(0, run.progress * 100))}%` }} /></div>}
            <div className="run-card-summary"><span>质心 {metric(run.summary?.com_z)} m</span><span>触及 {metric(run.summary?.steps_descended, 0)} 阶</span></div>
          </button>
        ))}
      </div>
      <div className="artifact-dock">
        <span className="eyebrow">结果导出</span>
        <div className="artifact-buttons">
          <button onClick={() => onExport("config.json")} disabled={!selectedId}>JSON</button>
          <button onClick={() => onExport("metrics.csv")} disabled={!selectedId}>CSV</button>
          <button onClick={() => onExport("trajectory.h5")} disabled={!selectedId}>HDF5</button>
        </div>
      </div>
    </aside>
  );
}

function ResearchSummary({ run }: { run: RunResult }) {
  const summary = run.summary || {};
  return (
    <section className="result-summary" data-testid="result-summary">
      <div className="result-summary-heading"><div><span className="eyebrow">研究指标</span><strong>结果诊断</strong></div><span className={`result-status result-status-${run.status}`}>{statusLabels[run.status] || run.status}</span></div>
      <div className="result-grid">
        <ResultMetric label="准备状态" value={prepareStatus(summary.prepare_status)} />
        <ResultMetric label="数值检查" value={validationStatus(summary.numerical_validation, "numerical")} />
        <ResultMetric label="实验支持" value={validationStatus(summary.experimental_validation, "experimental")} />
        <ResultMetric label="准备力残差" value={formatValue(summary.settle_force_residual, 4)} />
        <ResultMetric label="锚点误差" value={formatValue(summary.settle_anchor_error_m, 5, "m")} />
        <ResultMetric label="下端启动" value={formatTime(summary.bottom_onset_time)} />
        <ResultMetric label="收缩时间" value={formatTime(summary.collapse_time)} />
      </div>
      {run.config?.scenario === "stairs" && <StairSupportSummary summary={summary} />}
    </section>
  );
}

function StairSupportSummary({ summary }: { summary: RunSummary }) {
  const support = summary.support_diagnostics;
  const supportEvents = Array.isArray(support?.events) ? support.events : [];
  const legacyEventsAvailable = Array.isArray(summary.step_events);
  const legacyEvents = legacyEventsAvailable ? summary.step_events || [] : [];
  const firstTouchCount = support ? formatCount(supportEvents.filter((event) => Boolean(event.first_touch_label)).length, " 阶") : legacyEventsAvailable ? formatCount(legacyEvents.length, " 阶") : "未验证";
  const sustainedEndpointCount = support ? formatCount(supportEvents.filter(hasEndpointSupport).length, " 阶") : "未验证";
  const ambiguityCount = support ? formatCount(supportEvents.filter((event) => event.ambiguous).length, " 阶") : "未验证";
  const candidateCount = support ? formatCount(support.candidate_flip_count) : "未验证";
  const confirmedCount = support ? formatCount(support.confirmed_flip_count) : "未验证";
  const reasons = support ? uniqueReasons(supportEvents) : [];
  return (
    <div className="stairs-result">
      <div className="stairs-result-line"><span>运动分类</span><strong>{movementStatus(summary.movement_classification)}</strong></div>
      <div className="result-grid">
        <ResultMetric label="首次接触" value={firstTouchCount} />
        <ResultMetric label="持续端圈支撑" value={sustainedEndpointCount} />
        <ResultMetric label="端圈换阶支撑" value={support ? `${confirmedCount}（候选 ${candidateCount}）` : "未验证"} />
        <ResultMetric label="支撑歧义" value={ambiguityCount} />
      </div>
      {supportEvents.length > 0 ? <ul className="step-event-list">{supportEvents.map((event, index) => <li key={`${event.step ?? "step"}-${event.first_touch_time ?? index}`}>{formatSupportEvent(event, index)}</li>)}</ul> : legacyEvents.length > 0 ? <ul className="step-event-list">{legacyEvents.map((event, index) => <li key={`${event.step ?? "step"}-${event.time ?? index}`}>首次接触 · {formatStepEvent(event, index)}</li>)}</ul> : <div className="step-event-empty">释放后未检测到阶梯支撑事件</div>}
      {reasons.length > 0 && <div className="step-event-empty">歧义原因：{reasons.join("；")}</div>}
      <div className="step-event-empty">拱形首次落位需单独记数；完整翻转还需核查抬起、越端和落阶动作。</div>
    </div>
  );
}

function ResultMetric({ label, value }: { label: string; value: string }) {
  return <div className="result-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function prepareStatus(value: unknown): string {
  const normalized = normalize(value);
  if (normalized === "converged") return "已收敛";
  if (normalized === "not_converged") return "未收敛";
  if (normalized === "not_applicable") return "不适用";
  return "未验证";
}

function validationStatus(value: unknown, kind: "numerical" | "experimental"): string {
  const normalized = normalize(value);
  if (kind === "numerical") {
    if (normalized === "passed") return "通过";
    if (normalized === "failed") return "未通过";
    return "未验证";
  }
  if (normalized === "supported") return "获得支持";
  if (["not_supported", "unsupported"].includes(normalized)) return "暂无支持";
  return "未验证";
}

function movementStatus(value: unknown): string {
  const normalized = normalize(value);
  if (normalized === "flip") return "端圈交替换阶";
  if (normalized === "ambiguous_flip") return "换阶候选（支撑有歧义）";
  if (normalized === "sliding") return "滑移";
  if (normalized === "stopped") return "停止";
  if (normalized === "side_fall") return "侧落";
  if (normalized === "no_step_contact") return "未发生楼梯接触";
  return "未检测";
}

function formatSupportEvent(event: SupportEvent, index: number): string {
  const step = typeof event.step === "number" && Number.isFinite(event.step) ? `第${event.step}阶` : `事件${index + 1}`;
  const firstTouch = event.first_touch_label || "未检测";
  const sustained = event.first_sustained_label || "未检测";
  const support = endpointSupportText(event);
  const ambiguity = event.ambiguous ? `支撑歧义${event.ambiguity_reasons?.length ? `：${event.ambiguity_reasons.join("、")}` : ""}` : "支撑无歧义";
  return `${step} · 首次接触 ${firstTouch} · 持续端圈 ${sustained} · ${support} · ${ambiguity}`;
}

function hasEndpointSupport(event: SupportEvent): boolean {
  return Boolean(event.first_sustained_label === "first" || event.first_sustained_label === "last" || (event.endpoint_supports && event.endpoint_supports.length > 0));
}

function endpointSupportText(event: SupportEvent): string {
  const labels = (event.endpoint_supports || []).map((support) => support.label).filter((label): label is string => Boolean(label));
  return labels.length ? `端圈 ${labels.join("、")}` : "端圈未建立";
}

function uniqueReasons(events: SupportEvent[]): string[] {
  return [...new Set(events.flatMap((event) => event.ambiguity_reasons || []))];
}

function formatCount(value: unknown, suffix = ""): string {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.max(0, Math.round(value))}${suffix}` : "未验证";
}

function formatStepEvent(event: StepEvent, index: number): string {
  const step = typeof event.step === "number" && Number.isFinite(event.step) ? `第${event.step}阶` : `事件${index + 1}`;
  const end = event.end === "first" ? "材料首端" : event.end === "last" ? "材料末端" : event.end ? String(event.end) : "端点未分类";
  return `${step} · ${end} · ${formatTime(event.time)}`;
}

function formatTime(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(3)} s` : "未检测";
}

function formatValue(value: unknown, digits: number, unit = ""): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${unit ? ` ${unit}` : ""}` : "未验证";
}

function normalize(value: unknown): string {
  return value === undefined || value === null ? "" : String(value).trim().toLowerCase();
}

function formatDate(value: string): string {
  const stamp = Number(value);
  if (!Number.isFinite(stamp)) return value.slice(0, 8);
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(stamp * 1000));
}

function metric(value: number | undefined, digits = 3): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}
