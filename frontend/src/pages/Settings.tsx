import { useState } from "react";
import { api, type CloakConfig } from "../lib";
import { Card, CardTitle, Slider, Toggle } from "../components/ui";

export default function Settings({ cfg, setCfg, dark, setDark, onToast }: {
  cfg: CloakConfig | null; setCfg: (c: CloakConfig) => void;
  dark: boolean; setDark: (v: boolean) => void; onToast: (m: string) => void;
}) {
  const [rtsp, setRtsp] = useState("");

  const updateCfg = async (patch: Partial<CloakConfig>) => {
    if (!cfg) return;
    setCfg({ ...cfg, ...patch });
    try { await api("/config", { method: "POST", body: JSON.stringify(patch) }); }
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
        <Card label="Attack defaults" className="lg:col-span-2">
          <CardTitle>Attack defaults</CardTitle>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Slider label="EMA smoothing (new perturbation weight)" value={cfg.ema_alpha} min={0.1} max={1} step={0.05}
              onChange={(v) => updateCfg({ ema_alpha: v })} format={(v) => v.toFixed(2)} />
            <Slider label="Match threshold (cosine similarity)" value={cfg.enroll_threshold} min={0.2} max={0.9} step={0.05}
              onChange={(v) => updateCfg({ enroll_threshold: v })} format={(v) => v.toFixed(2)} />
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
