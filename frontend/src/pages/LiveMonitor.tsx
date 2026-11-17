import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Camera, CameraOff, Shield, ShieldOff } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useCloakSocket } from "../hooks/useCloakSocket";
import { api, qualityBadge, type CloakConfig } from "../lib";
import { Card, CardTitle, Gauge, InfoButton, Skeleton, Slider, Toast, Toggle } from "../components/ui";

export default function LiveMonitor({ cfg, setCfg, onToast }: {
  cfg: CloakConfig | null;
  setCfg: (c: CloakConfig) => void;
  onToast: (m: string) => void;
}) {
  const { connected, frame, error, connect, disconnect, sendFrame, clearError } = useCloakSocket();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [compare, setCompare] = useState(50); // split-slider position %
  const [compareMode, setCompareMode] = useState(false);
  const [showTour, setShowTour] = useState(() => !localStorage.getItem("cloak-tour-done"));
  const [history, setHistory] = useState<{ t: number; sim: number; fps: number; latency: number }[]>([]);

  const metrics = frame?.metrics;

  useEffect(() => {
    if (!metrics) return;
    setHistory((h) => [...h.slice(-59), {
      t: Date.now(), sim: metrics.similarity_after,
      fps: metrics.fps, latency: metrics.latency_ms,
    }]);
  }, [metrics]);

  // Webcam capture loop: draw to canvas, send JPEG ~10 fps.
  useEffect(() => {
    if (!cameraOn) return;
    let raf = 0; let last = 0;
    const loop = (t: number) => {
      raf = requestAnimationFrame(loop);
      if (t - last < 100 || !videoRef.current || !canvasRef.current) return;
      last = t;
      const v = videoRef.current, c = canvasRef.current;
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext("2d")!.drawImage(v, 0, 0);
      c.toBlob((b) => b && sendFrame(b), "image/jpeg", 0.85);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [cameraOn, sendFrame]);

  const startCamera = async () => {
    try {
      connect();
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      setCameraOn(true);
    } catch {
      onToast("Camera permission denied — allow webcam access or use the RTSP source in Settings.");
    }
  };
  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null; setCameraOn(false); disconnect();
  };

  const updateCfg = async (patch: Partial<CloakConfig>) => {
    if (!cfg) return;
    const next = { ...cfg, ...patch };
    setCfg(next);
    try { await api("/config", { method: "POST", body: JSON.stringify(patch) }); }
    catch { onToast("Backend unreachable — config change was not saved."); }
  };

  const img = (src?: string) => src ? `data:image/jpeg;base64,${src}` : undefined;

  return (
    <div className="space-y-4">
      {error && <Toast message={error} onClose={clearError} />}
      {showTour && (
        <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} role="dialog" aria-label="First-run tour"
          className="glass rounded-card p-4 text-sm">
          <p className="font-semibold">Welcome to Adversarial Cloak — 30-second tour</p>
          <ol className="mt-1 list-decimal space-y-1 pl-5 text-slate-600 dark:text-slate-300">
            <li>Press <b>Start camera</b> — frames stream to the backend over WebSocket.</li>
            <li>Watch <b>recognition confidence</b> fall on the Protected side while the two panels look identical. (With the real FaceNet/ArcFace weights it drops from ~98% to under 10%; the offline fallback nets show a smaller drop — see README.)</li>
            <li>Drag the <b>epsilon</b> slider: stronger cloak, slightly more visible noise.</li>
          </ol>
          <button onClick={() => { setShowTour(false); localStorage.setItem("cloak-tour-done", "1"); }}
            className="mt-3 rounded-card bg-teal-500 px-3 py-1.5 font-semibold text-white">Got it</button>
        </motion.div>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        {/* Video */}
        <Card label="Video comparison" className="lg:col-span-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle>Live monitor</CardTitle>
            <div className="flex items-center gap-2">
              <button onClick={() => setCompareMode(!compareMode)} aria-pressed={compareMode}
                className="rounded-card border border-slate-300 px-2.5 py-1 text-xs font-semibold dark:border-slate-600">
                {compareMode ? "Split compare: on" : "Split compare: off"}
              </button>
              <button onClick={cameraOn ? stopCamera : startCamera}
                className={`flex items-center gap-1.5 rounded-card px-3 py-1.5 text-sm font-semibold text-white ${cameraOn ? "bg-rose-500" : "bg-teal-500"}`}>
                {cameraOn ? <CameraOff size={16} /> : <Camera size={16} />} {cameraOn ? "Stop camera" : "Start camera"}
              </button>
            </div>
          </div>

          <video ref={videoRef} muted playsInline className="hidden" aria-hidden />
          <canvas ref={canvasRef} className="hidden" aria-hidden />

          {!frame ? (
            <div className="mt-4">
              {cameraOn ? <Skeleton className="aspect-video w-full" /> : (
                <div className="flex aspect-video flex-col items-center justify-center gap-2 rounded-card border-2 border-dashed border-slate-300 text-center dark:border-slate-600">
                  <Camera size={32} className="text-slate-400" />
                  <p className="font-medium">No video yet</p>
                  <p className="max-w-sm text-sm text-slate-500 dark:text-slate-400">Start your webcam to see the original and cloaked streams side by side. Nothing is recorded.</p>
                </div>
              )}
            </div>
          ) : (
            <div className="mt-4 space-y-3">
              {compareMode ? (
                <div className="relative aspect-video w-full overflow-hidden rounded-card" aria-label="Split comparison slider">
                  <img src={img(frame.protected)} alt="Protected video frame" className="absolute inset-0 h-full w-full object-cover" />
                  <div className="absolute inset-0 overflow-hidden" style={{ width: `${compare}%` }}>
                    <img src={img(frame.original)} alt="Original video frame" className="h-full w-full object-cover" style={{ width: `${100 / (compare / 100)}%`, maxWidth: "none" }} />
                  </div>
                  <input type="range" aria-label="Compare original and protected" min={5} max={95} value={compare}
                    onChange={(e) => setCompare(Number(e.target.value))}
                    className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0" />
                  <div className="pointer-events-none absolute top-0 h-full w-0.5 bg-white shadow" style={{ left: `${compare}%` }} />
                  <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs font-semibold text-white">Original</span>
                  <span className="absolute right-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs font-semibold text-white">Protected</span>
                </div>
              ) : (
                <div className="grid gap-3 sm:grid-cols-2">
                  {(["original", "protected"] as const).map((side) => (
                    <figure key={side} className="relative">
                      <img src={img(frame[side])} alt={`${side} video frame`} className="aspect-video w-full rounded-card object-cover" />
                      <figcaption className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs font-semibold text-white capitalize">{side}</figcaption>
                      {frame.boxes.map((b) => (
                        <div key={b.track_id} aria-hidden
                          className="absolute rounded border-2 border-teal-400"
                          style={{ left: `${(b.x / 640) * 100}%`, top: `${(b.y / 480) * 100}%`, width: `${(b.w / 640) * 100}%`, height: `${(b.h / 480) * 100}%` }} />
                      ))}
                    </figure>
                  ))}
                </div>
              )}
              <p className="text-xs text-slate-500 dark:text-slate-400" role="status">
                {connected ? "● Connected" : "○ Disconnected"} · {metrics?.faces ?? 0} face(s) · {metrics?.fps ?? 0} FPS · {metrics?.latency_ms ?? 0} ms
              </p>
            </div>
          )}
        </Card>

        {/* Gauges */}
        <Card label="Recognition confidence">
          <CardTitle>Recognition confidence</CardTitle>
          <div className="mt-2 flex justify-around">
            <Gauge value={metrics?.confidence_before ?? 0} label="Before" sub="original frame" />
            <Gauge value={metrics?.confidence_after ?? 0} label="After" sub="cloaked frame" />
          </div>
          <p className="mt-3 text-center text-sm">
            {metrics?.attack_success
              ? <span className="font-semibold text-emerald-600 dark:text-emerald-400">✓ Identity hidden — recogniser no longer matches</span>
              : metrics?.faces
                ? <span className="text-amber-600 dark:text-amber-400">Face detected, but similarity is still above the match threshold — raise epsilon or PGD steps.</span>
                : <span className="text-slate-500 dark:text-slate-400">Waiting for a face…</span>}
          </p>
          <div className="mt-4 grid grid-cols-3 gap-2 text-center">
            {[
              ["SSIM", metrics?.ssim ?? 1, "0.95", true],
              ["PSNR", metrics?.psnr ?? 99, "30 dB", true],
              ["LPIPS", metrics?.lpips ?? 0, "0.05", false],
            ].map(([label, v, good, higher]) => (
              <div key={label as string} className="rounded-card bg-slate-100/70 p-2 dark:bg-slate-800/60">
                <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
                <div className="text-lg font-bold tabular-nums">{typeof v === "number" ? v.toFixed(label === "PSNR" ? 1 : 3) : v}</div>
                <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${qualityBadge(v as number, Number(good), Number(good) * 0.9, higher as boolean)}`}>
                  {label === "SSIM" ? "near-identical" : label === "PSNR" ? "high fidelity" : "imperceptible"}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {/* Charts */}
        <Card label="Live charts" className="lg:col-span-2">
          <CardTitle>Similarity · FPS · Latency</CardTitle>
          {history.length < 2 ? <Skeleton className="mt-3 h-56 w-full" /> : (
            <div className="mt-2 h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="t" tickFormatter={(t) => new Date(t).toLocaleTimeString()} tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip labelFormatter={(t) => new Date(Number(t)).toLocaleTimeString()} />
                  <Legend />
                  <Line type="monotone" dataKey="sim" name="Similarity after cloak" stroke="#14b8a6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
          <div className="mt-2 grid grid-cols-3 gap-2 text-center text-sm">
            <div>FPS <b className="tabular-nums">{metrics?.fps ?? 0}</b></div>
            <div>Latency <b className="tabular-nums">{metrics?.latency_ms ?? 0} ms</b></div>
            <div>Similarity <b className="tabular-nums">{metrics?.similarity_after?.toFixed(3) ?? "—"}</b></div>
          </div>
        </Card>

        {/* Controls */}
        <Card label="Protection controls">
          <div className="flex items-center justify-between">
            <CardTitle>Control panel</CardTitle>
            <button onClick={() => updateCfg({ enabled: !cfg?.enabled })} aria-pressed={cfg?.enabled}
              className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-bold text-white ${cfg?.enabled ? "bg-teal-500" : "bg-slate-400"}`}>
              {cfg?.enabled ? <Shield size={15} /> : <ShieldOff size={15} />} {cfg?.enabled ? "Protected" : "Off"}
            </button>
          </div>
          {cfg ? (
            <div className="mt-4 space-y-4">
              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Strength</p>
              <Slider label="Epsilon (perturbation budget)" value={cfg.epsilon} min={2 / 255} max={16 / 255} step={1 / 255}
                onChange={(v) => updateCfg({ epsilon: v })} format={(v) => `${Math.round(v * 255)}/255`}
                info="How strong the invisible change is. Higher hides you better, but adds faint visible noise." />
              <Slider label="PGD steps" value={cfg.steps} min={1} max={20} step={1} onChange={(v) => updateCfg({ steps: v })}
                info="How many times the cloak is refined per face. More steps = stronger cloak, but slower." />
              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Speed</p>
              <Slider label="Update interval N (frames)" value={cfg.update_interval} min={1} max={15} step={1}
                onChange={(v) => updateCfg({ update_interval: v })} format={(v) => `every ${v} frames`}
                info="How often the cloak is recalculated. Higher = faster video, slightly weaker protection when you move." />
              <label className="block text-sm font-medium">Surrogate model
                <InfoButton label="Surrogate model" text="The AI model the cloak is tuned against. FaceNet is the standard choice; ArcFace is used to check the result." />
                <select aria-label="Surrogate model" value={cfg.surrogate} onChange={(e) => updateCfg({ surrogate: e.target.value })}
                  className="mt-1 w-full rounded-card border border-slate-300 bg-transparent px-2 py-1.5 dark:border-slate-600">
                  <option value="facenet">FaceNet (differentiable)</option>
                  <option value="arcface">ArcFace (evaluation)</option>
                </select>
              </label>
              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Robustness</p>
              <div className="flex items-center justify-between text-sm"><span>EOT robustness (JPEG / resize / blur)<InfoButton label="EOT robustness" text="Keeps the cloak working after the video is compressed, resized or blurred — like when a site re-encodes your video." /></span><Toggle checked={cfg.use_eot} onChange={(v) => updateCfg({ use_eot: v })} label="EOT robustness" /></div>
              <div className="flex items-center justify-between text-sm"><span>Facial-region mask (eyes / nose / mouth)<InfoButton label="Facial-region mask" text="Focuses the change on the eyes, nose and mouth, so the rest of your face stays untouched." /></span><Toggle checked={cfg.use_mask} onChange={(v) => updateCfg({ use_mask: v })} label="Facial-region mask" /></div>
              <p className="text-xs text-slate-500 dark:text-slate-400">Higher epsilon and more steps hide identity better but add visible noise. EOT keeps the cloak working after JPEG compression and resizing.</p>
            </div>
          ) : <Skeleton className="mt-4 h-64 w-full" />}
        </Card>
      </div>
    </div>
  );
}
