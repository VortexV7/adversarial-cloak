import { useEffect, useState } from "react";
import { Cpu, Gauge, Zap } from "lucide-react";
import { api, type CloakConfig, type DeviceStatus } from "../lib";
import { Card, CardTitle, InfoButton, Slider, Toggle } from "../components/ui";

const DEVICE_OPTIONS: { id: string; label: string; hint: string }[] = [
  { id: "auto", label: "Auto", hint: "GPU if one is available, otherwise CPU" },
  { id: "cpu", label: "CPU only", hint: "Works everywhere, slowest" },
  { id: "cuda", label: "GPU only", hint: "AI models run on the GPU — much faster" },
  { id: "both", label: "Both", hint: "GPU runs the AI, CPU handles video — fastest" },
];

export default function Settings({ cfg, setCfg, dark, setDark, onToast }: {
  cfg: CloakConfig | null; setCfg: (c: CloakConfig) => void;
  dark: boolean; setDark: (v: boolean) => void; onToast: (m: string) => void;
}) {
  const [rtsp, setRtsp] = useState("");
  const [device, setDevice] = useState<DeviceStatus | null>(null);

  useEffect(() => {
    api<DeviceStatus>("/device").then(setDevice).catch(() => undefined);
  }, []);

  const updateCfg = async (patch: Partial<CloakConfig>) => {
    if (!cfg) return;
    setCfg({ ...cfg, ...patch });
    try {
      await api("/config", { method: "POST", body: JSON.stringify(patch) });
      if (patch.device) {
        const status = await api<DeviceStatus>("/device");
        setDevice(status);
        if (status.fallback_warning) onToast("No GPU found on this machine — fell back to CPU.");
      }
    }
    catch { onToast("Backend unreachable — setting was not saved."); }
  };

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card label="Appearance">
        <CardTitle>Appearance</CardTitle>
        <div className="mt-4 flex items-center justify-between text-sm">
          <span>Dark theme</span>
          <Toggle checked={dark} onChange={setDark} label="Dark theme" />
        </div>
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">The dark theme is a deep navy/slate background with an electric teal accent; light mode uses the same accent on white.</p>
      </Card>

      <Card label="Camera source">
        <CardTitle>Camera source</CardTitle>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Browser webcam is the default. You can also point the backend at an RTSP / IP camera.</p>
        <label className="mt-3 block text-sm font-medium">RTSP / IP camera URL
          <input value={rtsp} onChange={(e) => setRtsp(e.target.value)} placeholder="rtsp://192.168.1.10:554/stream"
            aria-label="RTSP camera URL" className="mt-1 w-full rounded-card border border-slate-300 bg-transparent px-3 py-2 dark:border-slate-600" />
        </label>
        <div className="mt-3 flex gap-2">
          <button onClick={async () => { await api("/rtsp/start", { method: "POST", body: JSON.stringify({ url: rtsp }) }); onToast("RTSP source started — metrics update in the Live Monitor."); }}
            disabled={!rtsp} className="rounded-card bg-teal-500 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50">Start RTSP</button>
          <button onClick={async () => { await api("/rtsp/stop", { method: "POST" }); }} className="rounded-card border border-slate-300 px-3 py-1.5 text-sm font-semibold dark:border-slate-600">Stop</button>
        </div>
      </Card>

      {cfg && (
        <Card label="Performance — device" className="lg:col-span-2">
          <div className="flex items-center gap-2">
            <CardTitle>Performance — device</CardTitle>
            <InfoButton label="Device" text="Where the heavy AI work runs. A GPU makes the cloak much faster; without one, everything runs on the CPU." />
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {DEVICE_OPTIONS.map((opt) => {
              const active = cfg.device === opt.id;
              const unavailable = (opt.id === "cuda" || opt.id === "both") && device && !device.cuda_available;
              return (
                <button key={opt.id} onClick={() => updateCfg({ device: opt.id })} aria-pressed={active}
                  className={`rounded-card border p-3 text-left transition-colors ${active
                    ? "border-teal-500 bg-teal-500/10"
                    : "border-slate-300 hover:border-teal-400 dark:border-slate-600"}`}>
                  <span className="flex items-center gap-1.5 text-sm font-semibold">
                    {opt.id === "cpu" ? <Cpu size={15} /> : opt.id === "auto" ? <Gauge size={15} /> : <Zap size={15} />}
                    {opt.label}
                    {active && <span className="ml-auto rounded-full bg-teal-500 px-1.5 py-0.5 text-[10px] font-bold text-white">ON</span>}
                  </span>
                  <span className="mt-1 block text-xs text-slate-500 dark:text-slate-400">{opt.hint}</span>
                  {unavailable && <span className="mt-1 block text-xs font-medium text-amber-600 dark:text-amber-400">No GPU on this machine — will use CPU</span>}
                </button>
              );
            })}
          </div>
          <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
            {device?.cuda_available
              ? <>GPU detected: <b className="text-slate-700 dark:text-slate-200">{device.cuda_name}</b> — the AI models are running on it{device.using_gpu ? "" : " once you pick GPU or Both"}.</>
              : "No GPU detected on this machine — all four modes run on the CPU here. On a machine with an NVIDIA GPU, pick GPU or Both for the fastest processing."}
            {" "}Face detection always runs on the CPU (MediaPipe is CPU-only), so Both splits the work: GPU for the AI models, CPU for detection and video.
          </p>
        </Card>
      )}

      {cfg && (
        <Card label="Attack defaults" className="lg:col-span-2">
          <CardTitle>Attack defaults</CardTitle>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Slider label="EMA smoothing (new perturbation weight)" value={cfg.ema_alpha} min={0.1} max={1} step={0.05}
              onChange={(v) => updateCfg({ ema_alpha: v })} format={(v) => v.toFixed(2)}
              info="How much of the new cloak to mix in each time it is recalculated. Higher follows your movement faster; lower is smoother." />
            <Slider label="Match threshold (cosine similarity)" value={cfg.enroll_threshold} min={0.2} max={0.9} step={0.05}
              onChange={(v) => updateCfg({ enroll_threshold: v })} format={(v) => v.toFixed(2)}
              info="The score above which a face counts as 'the same person'. Below it, the cloak has worked." />
          </div>
          <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
            Honest performance note: on a laptop GPU the pipeline targets ≥15 FPS; on CPU expect roughly
            5–10 FPS at 960px width with the default 10 PGD steps (measured on this project's VM).
            Lower the PGD steps or raise the update interval N for more FPS.
          </p>
        </Card>
      )}
    </div>
  );
}
