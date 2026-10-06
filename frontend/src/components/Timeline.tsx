import type { Frame, RunStatus } from "../types";

interface Props {
  frames: Frame[];
  currentIndex: number;
  status: RunStatus | string;
  speed: number;
  onIndexChange: (index: number) => void;
  onAction: (action: "pause" | "resume" | "step" | "cancel") => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  isPlaying: boolean;
  onTogglePlayback: () => void;
}

export function Timeline({ frames, currentIndex, status, speed, onIndexChange, onAction, onReset, onSpeedChange, isPlaying, onTogglePlayback }: Props) {
  const playable = status === "running" || status === "paused" || status === "completed" || status === "failed" || status === "cancelled";
  const playing = isPlaying || status === "running";
  const current = frames[currentIndex];
  return (
    <div className="timeline-card" data-testid="timeline">
      <div className="timeline-topline">
        <span className="eyebrow">回放控制</span>
        <span className="timeline-time">{current ? `${current.time.toFixed(3)} / ${frames[frames.length - 1]?.time.toFixed(3) || "0.000"} s` : "尚无物理帧"}</span>
      </div>
      <input className="timeline-range" type="range" min={0} max={Math.max(frames.length - 1, 0)} value={Math.min(currentIndex, Math.max(frames.length - 1, 0))} onChange={(event) => onIndexChange(Number(event.target.value))} disabled={!frames.length} />
      <div className="timeline-controls">
        <button className="square-button" title="重置" onClick={onReset}>↺</button>
        <button className="play-button" disabled={!playable} onClick={() => { if (status === "completed" || status === "failed" || status === "cancelled") onTogglePlayback(); else onAction(playing ? "pause" : "resume"); }}><span>{playing ? "Ⅱ" : "▶"}</span>{playing ? "暂停" : "开始"}</button>
        <button className="square-button" disabled={!playable || playing} title="单步" onClick={() => onAction("step")}>›</button>
        <button className="square-button danger" disabled={!playable || status === "completed" || status === "cancelled"} title="取消" onClick={() => onAction("cancel")}>×</button>
        <div className="speed-control"><span>播放</span>{[0.25, 0.5, 1, 2].map((value) => <button key={value} className={speed === value ? "active" : ""} onClick={() => onSpeedChange(value)}>{value}×</button>)}</div>
      </div>
    </div>
  );
}
