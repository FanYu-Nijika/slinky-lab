import type { RunResult, RunStatus } from "../types";

interface Props {
  runs: RunResult[];
  selectedId?: string;
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

export function RunHistory({ runs, selectedId, onSelect, onExport }: Props) {
  return (
    <aside className="panel right-panel" data-testid="run-history">
      <div className="panel-heading">
        <div><span className="eyebrow">实验记录</span><h2>运行历史</h2></div>
        <span className="panel-count">{runs.length.toString().padStart(2, "0")}</span>
      </div>
      <div className="panel-scroll history-scroll">
        {runs.length === 0 && <div className="empty-history"><div className="empty-orbit">◌</div><strong>还没有运行记录</strong><span>从左侧参数开始一次研究运行</span></div>}
        {runs.map((run) => (
          <button key={run.run_id} className={`run-card ${selectedId === run.run_id ? "selected" : ""}`} onClick={() => onSelect(run)}>
            <div className="run-card-top"><span className={`status-dot status-${run.status}`} /><span className="run-name">{run.config?.name || run.run_id.slice(0, 8)}</span><span className={`status-label status-text-${run.status}`}>{statusLabels[run.status] || run.status}</span></div>
            <div className="run-card-meta"><span>{scenarioLabel(run)}</span><span>{formatDate(run.run_id)}</span></div>
            {run.progress !== undefined && run.status === "running" && <div className="run-progress"><i style={{ width: `${Math.min(100, Math.max(0, run.progress * 100))}%` }} /></div>}
            <div className="run-card-summary"><span>质心 {metric(run.summary?.com_z)} m</span><span>{metric(run.summary?.steps_descended, 0)} 阶</span></div>
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

function formatDate(value: string): string {
  const stamp = Number(value);
  if (!Number.isFinite(stamp)) return value.slice(0, 8);
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(stamp * 1000));
}

function metric(value: number | undefined, digits = 3): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}
