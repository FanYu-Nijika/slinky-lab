import type {
  Frame,
  PresetInfo,
  RenderMetadata,
  RunConfig,
  RunResult,
  SweepConfig,
  SweepResult,
} from "./types";

const API_PREFIX = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || body.error || detail;
    } catch {
      // The status text is enough when the service returns a non-JSON error.
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  return parseResponse<T>(response);
}

export async function getHealth(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/health");
}

export async function getPresets(): Promise<PresetInfo[]> {
  const body = await request<PresetInfo[] | { presets: PresetInfo[] }>("/presets");
  return Array.isArray(body) ? body : body.presets || [];
}

export async function getRuns(): Promise<RunResult[]> {
  const body = await request<RunResult[] | { runs: RunResult[] }>("/runs");
  return Array.isArray(body) ? body : body.runs || [];
}

export async function createRun(config: RunConfig): Promise<{ run_id: string; status: string }> {
  return request<{ run_id: string; status: string }>("/runs", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function getRun(runId: string): Promise<RunResult> {
  return request<RunResult>(`/runs/${encodeURIComponent(runId)}`);
}

export async function getFrames(runId: string, start = 0, limit = 1000): Promise<{ frames: Frame[]; total: number; metadata?: RenderMetadata }> {
  const boundedLimit = Math.min(Math.max(limit, 1), 1000);
  return request<{ frames: Frame[]; total: number; metadata?: RenderMetadata }>(
    `/runs/${encodeURIComponent(runId)}/frames?start=${start}&limit=${boundedLimit}`,
  );
}

export async function getAllFrames(runId: string): Promise<{ frames: Frame[]; total: number; metadata?: RenderMetadata }> {
  const frames: Frame[] = [];
  let total = 0;
  let metadata: RenderMetadata | undefined;
  do {
    const page = await getFrames(runId, frames.length, 1000);
    frames.push(...(page.frames || []));
    total = page.total || frames.length;
    metadata = metadata || page.metadata;
    if (!page.frames?.length) break;
  } while (frames.length < total);
  return { frames, total, metadata };
}

export async function sendCommand(runId: string, action: "pause" | "resume" | "step" | "cancel"): Promise<RunResult> {
  return request<RunResult>(`/runs/${encodeURIComponent(runId)}/commands`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });
}

export function artifactUrl(runId: string, name: string): string {
  return `${API_PREFIX}/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(name)}`;
}

export async function downloadArtifact(runId: string, name: string): Promise<Blob> {
  const response = await fetch(artifactUrl(runId, name));
  if (!response.ok) throw new ApiError(`无法下载 ${name}`, response.status);
  return response.blob();
}

export async function createSweep(config: SweepConfig): Promise<{ sweep_id: string; status: string }> {
  return request<{ sweep_id: string; status: string }>("/sweeps", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function getSweep(sweepId: string): Promise<SweepResult> {
  return request<SweepResult>(`/sweeps/${encodeURIComponent(sweepId)}`);
}

export async function uploadReference(runId: string, file: File, metric: string): Promise<RunResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("metric", metric);
  const response = await fetch(`${API_PREFIX}/runs/${encodeURIComponent(runId)}/reference`, { method: "POST", body: form });
  return parseResponse<RunResult>(response);
}

export async function getReference(runId: string): Promise<{ metric: string; time: number[]; value: number[]; comparison?: Record<string, unknown> }> {
  return request<{ metric: string; time: number[]; value: number[]; comparison?: Record<string, unknown> }>(`/runs/${encodeURIComponent(runId)}/reference`);
}

export function streamUrl(runId: string): string {
  const base = API_PREFIX.startsWith("http") ? API_PREFIX : window.location.origin + API_PREFIX;
  const wsBase = base.replace(/^http/, "ws");
  return `${wsBase}/runs/${encodeURIComponent(runId)}/stream`;
}
