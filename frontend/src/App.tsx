import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChartsPanel } from "./components/ChartsPanel";
import { ParameterPanel } from "./components/ParameterPanel";
import { RunHistory } from "./components/RunHistory";
import { SceneView } from "./components/SceneView";
import { SweepPanel } from "./components/SweepPanel";
import { Timeline } from "./components/Timeline";
import { ApiError, createRun, createSweep, downloadArtifact, getAllFrames, getFrames, getPresets, getRun, getRuns, getSweep, sendCommand, streamUrl, uploadReference } from "./api";
import { makePreviewFrame, makePreviewMetadata } from "./preview";
import type { Frame, MetricKey, PresetInfo, RenderMetadata, RunConfig, RunResult, RunStatus, SweepResult } from "./types";
import { DEFAULT_CONFIG, DEFAULT_DROP_VALIDATION_CONFIG } from "./types";

const ACTIVE_STATUSES = new Set(["queued", "preparing", "running", "paused"]);

function cloneConfig(config: RunConfig): RunConfig {
  return JSON.parse(JSON.stringify(config)) as RunConfig;
}

export default function App() {
  const [config, setConfig] = useState<RunConfig>(() => cloneConfig(DEFAULT_DROP_VALIDATION_CONFIG));
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [runs, setRuns] = useState<RunResult[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunResult | null>(null);
  const [selectedId, setSelectedId] = useState<string>();
  const [status, setStatus] = useState<RunStatus | string>("preparing");
  const [frames, setFrames] = useState<Frame[]>([]);
  const [metadata, setMetadata] = useState<RenderMetadata>();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [online, setOnline] = useState(true);
  const [loading, setLoading] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState("");
  const [showContacts, setShowContacts] = useState(false);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [geometryMode, setGeometryMode] = useState<"boxes" | "smooth">("smooth");
  const [cameraView, setCameraView] = useState<"orbit" | "front" | "side" | "top">("orbit");
  const [followCamera, setFollowCamera] = useState(true);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [sweep, setSweep] = useState<SweepResult | null>(null);
  const [sweepId, setSweepId] = useState<string>();
  const socketRef = useRef<WebSocket>();
  const pollRef = useRef<number>();
  const followLiveRef = useRef(true);
  const playbackTimerRef = useRef<number>();
  const generationRef = useRef(0);
  const configEditedRef = useRef(false);

  const currentFrame = frames[currentIndex] || makePreviewFrame(config, 0);
  const previewMetadata = useMemo(() => makePreviewMetadata(config), [config]);
  const renderMetadata = metadata || previewMetadata;
  const engineVersion = metadata?.engine_version || metadata?.mujoco_version;
  const hasPhysicalResult = Boolean(selectedId && frames.length && engineVersion && engineVersion !== "none");

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([getPresets(), getRuns()]).then((results) => {
      if (cancelled) return;
      const presetResult = results[0];
      const runsResult = results[1];
      if (presetResult.status === "fulfilled") {
        setPresets(presetResult.value);
        const validationPreset = presetResult.value.find((item) => item.id === "drop-validation" && item.config);
        if (!configEditedRef.current && validationPreset?.config) {
          setConfig(cloneConfig(validationPreset.config));
          setDirty(false);
        }
      } else {
        setOnline(false);
        if (!configEditedRef.current) {
          setConfig(cloneConfig(DEFAULT_DROP_VALIDATION_CONFIG));
          setDirty(false);
        }
      }
      if (runsResult.status === "fulfilled") setRuns(runsResult.value);
      else setOnline(false);
    });
    return () => { cancelled = true; };
  }, []);

  const applyRun = useCallback((run: RunResult, loadedFrames?: Frame[], loadedMetadata?: RenderMetadata) => {
    configEditedRef.current = true;
    setSelectedRun(run);
    setSelectedId(run.run_id);
    setConfig(cloneConfig(run.config));
    setStatus(run.status);
    setDirty(false);
    if (loadedFrames) {
      setFrames(loadedFrames);
      setCurrentIndex(Math.max(0, loadedFrames.length - 1));
    }
    setMetadata(loadedMetadata);
  }, []);

  const refreshRun = useCallback(async (runId: string, loadTrajectory = false, generation = generationRef.current) => {
    const run = await getRun(runId);
    if (generation !== generationRef.current) return;
    setSelectedRun(run);
    setStatus(run.status);
    setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]);
    if (loadTrajectory || !frames.length) {
      const frameResponse = await getAllFrames(runId);
      if (generation !== generationRef.current) return;
      setFrames(frameResponse.frames || []);
      if (frameResponse.metadata) setMetadata(frameResponse.metadata);
      setCurrentIndex(Math.max(0, (frameResponse.frames || []).length - 1));
    }
  }, [frames.length]);

  const appendFrame = useCallback((frame: Frame) => {
    setFrames((current) => {
      const existing = current.findIndex((item) => Math.abs(item.time - frame.time) < 1e-9);
      const next = existing >= 0 ? current.map((item, index) => index === existing ? frame : item) : [...current, frame].sort((a, b) => a.time - b.time);
      if (followLiveRef.current) setCurrentIndex(next.length - 1);
      return next;
    });
  }, []);

  const openStream = useCallback((runId: string, generation = generationRef.current) => {
    socketRef.current?.close();
    try {
      const socket = new WebSocket(streamUrl(runId));
      socket.onopen = () => { if (generation === generationRef.current) setOnline(true); };
      socket.onmessage = (event) => {
        if (generation !== generationRef.current) return;
        try {
          const message = JSON.parse(event.data) as { type: string; data?: unknown };
          const data = (message.data || {}) as Record<string, unknown>;
          if (message.type === "status") {
            const nextStatus = (data.status || message.data) as string;
            if (nextStatus) setStatus(nextStatus);
          } else if (message.type === "metadata") {
            setMetadata((data.metadata || data) as RenderMetadata);
          } else if (message.type === "frame") {
            appendFrame((data.frame || data) as Frame);
          } else if (message.type === "result") {
            const result = (data.result || data) as RunResult;
            setSelectedRun(result);
            setStatus(result.status);
            if (result.summary) setRuns((current) => current.map((item) => item.run_id === result.run_id ? result : item));
          } else if (message.type === "error") {
            setError(String(data.error || data.message || "计算服务返回错误"));
          }
        } catch {
          setError("无法解析计算服务的实时消息");
        }
      };
      socket.onerror = () => { if (generation === generationRef.current) setOnline(false); };
      socket.onclose = () => { if (generation === generationRef.current) socketRef.current = undefined; };
      socketRef.current = socket;
    } catch {
      setOnline(false);
    }
  }, [appendFrame]);

  useEffect(() => () => { socketRef.current?.close(); if (pollRef.current) window.clearInterval(pollRef.current); }, []);

  useEffect(() => {
    if (!selectedId || !ACTIVE_STATUSES.has(status)) return undefined;
    const generation = generationRef.current;
    let disposed = false;
    const poll = async () => {
      try {
        const run = await getRun(selectedId);
        if (disposed || generation !== generationRef.current) return;
        setSelectedRun(run);
        setStatus(run.status);
        setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]);
        if (run.metadata) setMetadata(run.metadata);
        if (run.status === "completed" || run.status === "failed" || run.status === "cancelled") {
          const frameResponse = await getAllFrames(selectedId);
          if (!disposed && generation === generationRef.current) {
            setFrames(frameResponse.frames || []);
            if (frameResponse.metadata) setMetadata(frameResponse.metadata);
            setCurrentIndex(Math.max(0, (frameResponse.frames || []).length - 1));
          }
        }
      } catch (cause) {
        if (!disposed && !(cause instanceof ApiError && cause.status === 404)) setOnline(false);
      }
    };
    void poll();
    pollRef.current = window.setInterval(() => void poll(), 1500);
    return () => { disposed = true; if (pollRef.current) window.clearInterval(pollRef.current); };
  }, [selectedId, status]);

  useEffect(() => {
    if (!isPlaying || frames.length < 2) return undefined;
    if (currentIndex >= frames.length - 1) {
      setIsPlaying(false);
      return undefined;
    }
    const currentTime = frames[currentIndex]?.time ?? 0;
    const nextTime = frames[currentIndex + 1]?.time ?? currentTime;
    const delay = Math.max(8, Math.min(1000, ((nextTime - currentTime) * 1000) / speed));
    playbackTimerRef.current = window.setTimeout(() => setCurrentIndex((index) => Math.min(index + 1, frames.length - 1)), delay);
    return () => { if (playbackTimerRef.current) window.clearTimeout(playbackTimerRef.current); };
  }, [isPlaying, speed, currentIndex, frames]);

  const changeConfig = (next: RunConfig) => {
    configEditedRef.current = true;
    setConfig(next);
    setDirty(true);
    setError("");
    followLiveRef.current = false;
  };

  const choosePreset = (idOrName: string) => {
    const preset = presets.find((item) => item.id === idOrName || item.name === idOrName);
    if (preset?.config) {
      changeConfig(cloneConfig(preset.config));
      return;
    }
    const next = cloneConfig(DEFAULT_CONFIG);
    next.scenario = idOrName === "stairs" ? "stairs" : "drop";
    next.name = next.scenario === "stairs" ? "翻转下楼梯实验" : "悬挂下落实验";
    changeConfig(next);
  };

  const handleRun = async () => {
    configEditedRef.current = true;
    const generation = ++generationRef.current;
    setLoading(true);
    setError("");
    setIsPlaying(false);
    setFrames([]);
    setMetadata(undefined);
    setSelectedRun(null);
    setSelectedId(undefined);
    setCurrentIndex(0);
    followLiveRef.current = true;
    try {
      const response = await createRun(config);
      const optimistic: RunResult = { run_id: response.run_id, status: response.status, config: cloneConfig(config), summary: {}, artifacts: [] };
      setRuns((current) => [optimistic, ...current.filter((item) => item.run_id !== optimistic.run_id)]);
      applyRun(optimistic, [], undefined);
      openStream(response.run_id, generation);
      try { await refreshRun(response.run_id, true, generation); } catch { /* stream/poll will hydrate a just-created run */ }
    } catch (cause) {
      setOnline(false);
      setError(cause instanceof Error ? cause.message : "无法创建运行");
    } finally {
      setLoading(false);
    }
  };

  const handleCommand = async (action: "pause" | "resume" | "step" | "cancel") => {
    if (!selectedId) return;
    try {
      const result = await sendCommand(selectedId, action);
      setSelectedRun(result);
      setStatus(result.status);
      if (action === "step") {
      const response = await getFrames(selectedId, frames.length, 4);
        response.frames.forEach(appendFrame);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "命令发送失败");
    }
  };

  const selectRun = async (run: RunResult) => {
    const generation = ++generationRef.current;
    setError("");
    try {
      const detail = await getRun(run.run_id);
      const response = await getAllFrames(run.run_id);
      if (generation !== generationRef.current) return;
      applyRun(detail, response.frames || [], response.metadata || detail.metadata);
      openStream(run.run_id, generation);
    } catch {
      if (generation !== generationRef.current) return;
      applyRun(run, run.frames || [], run.metadata);
      setOnline(false);
    }
  };

  const exportArtifact = async (name: string) => {
    if (!selectedId) return;
    try {
      const blob = await downloadArtifact(selectedId, name);
      saveBlob(blob, name);
    } catch {
      setError(`服务端尚未提供 ${name}，未生成本地替代文件`);
    }
  };

  const uploadReferenceFile = async (file: File, metric: MetricKey) => {
    if (!selectedId) return;
    try {
      const run = await uploadReference(selectedId, file, metric);
      setSelectedRun(run);
      setRuns((current) => current.map((item) => item.run_id === run.run_id ? run : item));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "参考数据上传失败");
    }
  };

  const startSweep = async (sweepConfig: { name: string; base: RunConfig; axes: Array<{ parameter: string; values: number[] }> }) => {
    const response = await createSweep(sweepConfig);
    setSweepId(response.sweep_id);
  };

  useEffect(() => {
    if (!sweepId) return undefined;
    let disposed = false;
    const poll = async () => {
      try {
        const result = await getSweep(sweepId);
        if (!disposed) setSweep(result);
      } catch (cause) {
        if (!disposed) setError(cause instanceof Error ? cause.message : "扫描状态读取失败");
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 1500);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [sweepId]);

  const statusLabel = ({ preparing: "准备模型", queued: "排队中", running: "计算中", paused: "已暂停", completed: "已完成", failed: "失败", cancelled: "已取消", interrupted: "中断" } as Record<string, string>)[status] || status;
  const progress = selectedRun?.progress;
  const headline = selectedRun?.config?.name || config.name;
  const modelVersionLabel = metadata?.model_version && engineVersion
    ? `模型 ${metadata.model_version} · 引擎 ${engineVersion}`
    : metadata?.model_version
      ? `模型 ${metadata.model_version}`
      : engineVersion
        ? `引擎 ${engineVersion}`
        : "模型 预览";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><span /><span /><span /></div><div><strong>Slinky<span>Lab</span></strong><small>三维动力学研究工作台</small></div></div>
        <div className="topbar-center"><span className={`connection ${online ? "online" : "offline"}`}><i />{online ? "计算服务在线" : "离线预览"}</span><span className="separator" /><span className="experiment-name">{headline}{dirty && <em>未保存修改</em>}</span></div>
        <div className="topbar-actions"><button className="ghost-button" onClick={() => setLeftCollapsed((value) => !value)}>{leftCollapsed ? "显示参数" : "隐藏参数"}</button><button className="ghost-button" onClick={() => setRightCollapsed((value) => !value)}>{rightCollapsed ? "显示历史" : "隐藏历史"}</button><button className="primary-button" data-testid="start-run" disabled={loading} onClick={() => void handleRun()}><span>▶</span>{loading ? "提交中…" : "开始新运行"}</button></div>
      </header>

      <main className={`workspace ${leftCollapsed ? "left-hidden" : ""} ${rightCollapsed ? "right-hidden" : ""}`}>
        <ParameterPanel config={config} presets={presets} onChange={changeConfig} onPreset={choosePreset} collapsed={leftCollapsed} />
        <section className="center-column">
          <div className="center-heading"><div><span className="eyebrow">{config.scenario === "stairs" ? "SCENARIO / STAIRS" : "SCENARIO / DROP"}</span><h1>{config.scenario === "stairs" ? "翻转下楼梯" : "悬挂下落"}<span className="model-pill">{config.numerics.profile === "fine" ? "FINE" : "PREVIEW"}</span></h1></div><div className="view-toggles"><button title="由真实帧位置与截面姿态插值的矩形带面" className={geometryMode === "smooth" ? "active" : ""} onClick={() => setGeometryMode("smooth")}>⌁ 彩虹带</button><button title="显示 MuJoCo 真实碰撞盒体" className={geometryMode === "boxes" ? "active" : ""} onClick={() => setGeometryMode("boxes")}>▦ 碰撞几何</button><button className={showContacts ? "active" : ""} onClick={() => setShowContacts((value) => !value)}>⊙ 接触点</button><button className={showTrajectory ? "active" : ""} onClick={() => setShowTrajectory((value) => !value)}>⌁ 轨迹</button><button title="沿质心平移相机，保持下落主体可见" className={followCamera ? "active" : ""} onClick={() => setFollowCamera((value) => !value)}>◎ 跟随质心</button></div></div>
          <SceneView config={config} frame={currentFrame} metadata={renderMetadata} runKey={selectedId || "preview"} showContacts={showContacts} showTrajectory={showTrajectory} geometryMode={geometryMode} cameraView={cameraView} followCamera={followCamera} onCameraViewChange={setCameraView} />
          <Timeline frames={frames} currentIndex={currentIndex} status={status} hasRun={Boolean(selectedId)} speed={speed} onIndexChange={(index) => { followLiveRef.current = false; setCurrentIndex(index); }} onAction={(action) => void handleCommand(action)} onReset={() => { followLiveRef.current = false; setCurrentIndex(0); setIsPlaying(false); }} onSpeedChange={setSpeed} isPlaying={isPlaying} onTogglePlayback={() => setIsPlaying((value) => !value)} />
          <ChartsPanel frames={frames} run={selectedRun} onUploadReference={uploadReferenceFile} onExportPng={(dataUrl) => saveDataUrl(dataUrl, "slinky-chart.png")} />
          <SweepPanel base={config} sweep={sweep} onStart={startSweep} />
        </section>
        <RunHistory runs={runs} selectedId={selectedId} selectedRun={selectedRun} onSelect={(run) => void selectRun(run)} onExport={(name) => void exportArtifact(name)} />
      </main>

      <div className="statusbar"><div className="status-main"><span className={`status-dot status-${status}`} /><strong>{statusLabel}</strong>{progress !== undefined && ACTIVE_STATUSES.has(status) && <span className="progress-label">{Math.round(progress * 100)}%</span>}<span className="status-message">{error || (hasPhysicalResult ? "物理结果已加载，可拖动时间轴回放" : "调整参数后开始一次物理运行；当前为几何预览")}</span></div><div className="status-meta"><span>SI · Z-up</span><span data-testid="model-version">{modelVersionLabel}</span></div></div>
    </div>
  );
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function saveDataUrl(dataUrl: string, filename: string) {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  link.click();
}
