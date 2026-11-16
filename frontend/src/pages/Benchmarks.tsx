import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../lib";
import { Card, CardTitle, Skeleton } from "../components/ui";

interface Row {
  method: string; dataset: string; asr_facenet: number; asr_arcface: number;
  rank1_drop: number; ssim: number; psnr: number; lpips: number;
  robust_jpeg70_asr: number; robust_resize_asr: number; robust_blur_asr: number; latency_ms: number;
}

export default function Benchmarks({ onToast }: { onToast: (m: string) => void }) {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      const r = await api<{ status: string; rows?: Row[]; stderr?: string }>("/benchmark", { method: "POST" });
      if (r.status !== "ok" || !r.rows) throw new Error(r.stderr ?? "benchmark failed");
      setRows(r.rows);
    } catch (e) {
      onToast("Benchmark failed — run `python -m eval.run_eval` in the backend for the full log.");
    } finally { setBusy(false); }
  };

  return (
    <div className="space-y-4">
      <Card label="Benchmarks">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle>Benchmarks — FGSM vs PGD vs Ours</CardTitle>
            <p className="mt-1 max-w-2xl text-sm text-slate-500 dark:text-slate-400">
              Runs <code>eval/run_eval.py</code> on LFW pairs (or synthetic pairs when LFW is not downloaded).
              Attack success rate is measured at a fixed false-accept rate against FaceNet (white-box) and ArcFace (black-box transfer).
            </p>
          </div>
          <button onClick={run} disabled={busy} className="rounded-card bg-teal-500 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
            {busy ? "Running eval…" : "Run benchmark"}
          </button>
        </div>
        {busy && <Skeleton className="mt-4 h-64 w-full" />}
        {!rows && !busy && (
          <div className="mt-4 rounded-card border-2 border-dashed border-slate-300 p-6 text-center text-sm text-slate-500 dark:border-slate-600 dark:text-slate-400">
            No benchmark results yet. Press “Run benchmark” — the eval takes a few minutes on CPU.
          </div>
        )}
      </Card>

      {rows && (
        <>
          <Card label="Method comparison chart">
            <CardTitle>Attack success rate by method</CardTitle>
            <div className="mt-2 h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rows}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="method" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="asr_facenet" name="FaceNet (white-box)" fill="#14b8a6" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="asr_arcface" name="ArcFace (black-box)" fill="#0ea5e9" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card label="Results table">
            <CardTitle>Results table ({rows[0]?.dataset})</CardTitle>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead><tr className="text-slate-500 dark:text-slate-400">
                  {["Method", "ASR FaceNet", "ASR ArcFace", "Rank-1 drop", "SSIM", "PSNR", "LPIPS", "JPEG70 ASR", "Resize ASR", "Blur ASR", "Latency"].map((h) => (
                    <th key={h} scope="col" className="px-2 py-1 font-semibold">{h}</th>
                  ))}
                </tr></thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.method} className="border-t border-slate-200 dark:border-slate-700">
                      <td className="px-2 py-1.5 font-semibold">{r.method}</td>
                      <td className="px-2 py-1.5 tabular-nums">{(r.asr_facenet * 100).toFixed(1)}%</td>
                      <td className="px-2 py-1.5 tabular-nums">{(r.asr_arcface * 100).toFixed(1)}%</td>
                      <td className="px-2 py-1.5 tabular-nums">{r.rank1_drop.toFixed(3)}</td>
                      <td className="px-2 py-1.5 tabular-nums">{r.ssim.toFixed(3)}</td>
                      <td className="px-2 py-1.5 tabular-nums">{r.psnr.toFixed(1)}</td>
                      <td className="px-2 py-1.5 tabular-nums">{r.lpips.toFixed(3)}</td>
                      <td className="px-2 py-1.5 tabular-nums">{(r.robust_jpeg70_asr * 100).toFixed(1)}%</td>
                      <td className="px-2 py-1.5 tabular-nums">{(r.robust_resize_asr * 100).toFixed(1)}%</td>
                      <td className="px-2 py-1.5 tabular-nums">{(r.robust_blur_asr * 100).toFixed(1)}%</td>
                      <td className="px-2 py-1.5 tabular-nums">{r.latency_ms} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
