export interface Metrics {
  faces: number;
  similarity_before: number;
  similarity_after: number;
  confidence_before: number;
  confidence_after: number;
  attack_success: boolean;
  ssim: number;
  psnr: number;
  lpips: number;
  latency_ms: number;
  fps: number;
  similarity_history: number[];
  enabled: boolean;
  epsilon: number;
  steps: number;
  update_interval: number;
  surrogate: string;
  evaluator: string;
}

export interface CloakConfig {
  epsilon: number;
  steps: number;
  update_interval: number;
  ema_alpha: number;
  surrogate: string;
  evaluator: string;
  use_eot: boolean;
  use_mask: boolean;
  enabled: boolean;
  enroll_threshold: number;
  device: string; // auto | cpu | cuda | both
}

export interface DeviceStatus {
  mode: string;
  cuda_available: boolean;
  cuda_name: string | null;
  compute_device: string;
  detector_device: string;
  gpu_requested: boolean;
  using_gpu: boolean;
  fallback_warning: boolean;
}

export interface CloakImageResult {
  original: string;
  image: string;
  faces: number;
  boxes: { x: number; y: number; w: number; h: number; track_id: number; score: number }[];
  similarity_before: number;
  similarity_after: number;
  confidence_before: number;
  confidence_after: number;
  attack_success: boolean;
  ssim: number;
  psnr: number;
  lpips: number;
  latency_ms: number;
  device: string;
  device_mode: string;
}

export interface FrameMessage {
  type: "frame";
  original: string;
  protected: string;
  metrics: Metrics;
  boxes: { x: number; y: number; w: number; h: number; track_id: number; score: number }[];
}

export const API = (path: string) => `/api${path}`;
export const WS_URL =
  (import.meta.env.VITE_WS_URL as string | undefined) ??
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000/ws/stream`;

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API(path), {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export function qualityBadge(value: number, good: number, warn: number, higherIsBetter = true) {
  const ok = higherIsBetter ? value >= good : value <= good;
  const mid = higherIsBetter ? value >= warn : value <= warn;
  return ok ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
    : mid ? "bg-amber-500/15 text-amber-600 dark:text-amber-400"
    : "bg-rose-500/15 text-rose-600 dark:text-rose-400";
}
