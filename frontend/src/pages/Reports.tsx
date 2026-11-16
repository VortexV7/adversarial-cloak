import { useState } from "react";
import { Download, FileText } from "lucide-react";
import { Card, CardTitle } from "../components/ui";

export default function Reports({ onToast }: { onToast: (m: string) => void }) {
  const [busy, setBusy] = useState(false);
  const download = async (fmt: "csv" | "pdf") => {
    setBusy(true);
    try {
      const res = await fetch(`/api/export-report?fmt=${fmt}`, { method: "POST" });
      if (!res.ok) throw new Error(String(res.status));
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = fmt === "pdf" ? "cloak-report.pdf" : "cloak-report.csv";
      a.click();
    } catch {
      onToast(fmt === "pdf" ? "PDF export needs reportlab on the backend — CSV works without it." : "Report export failed — is the backend running?");
    } finally { setBusy(false); }
  };

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card label="Export a report">
        <CardTitle>Export a session report</CardTitle>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Exports the latest live-session metrics (similarity, confidence, SSIM / PSNR / LPIPS, latency, FPS)
          plus the current configuration, for the project report.
        </p>
        <div className="mt-4 flex gap-2">
          <button onClick={() => download("csv")} disabled={busy} className="flex items-center gap-2 rounded-card bg-teal-500 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
            <Download size={16} /> Download CSV
          </button>
          <button onClick={() => download("pdf")} disabled={busy} className="flex items-center gap-2 rounded-card border border-slate-300 px-4 py-2 text-sm font-semibold dark:border-slate-600 disabled:opacity-50">
            <FileText size={16} /> Download PDF
          </button>
        </div>
      </Card>
      <Card label="What to include in the report">
        <CardTitle>Report checklist</CardTitle>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600 dark:text-slate-300">
          <li>Threat model — see <code>docs/THREAT_MODEL.md</code> in the repo.</li>
          <li>Benchmark table from the Benchmarks page (<code>eval/results.csv</code>).</li>
          <li>Ablation plots: <code>eval/plots/ablation.png</code> and <code>eval/plots/methods.png</code>.</li>
          <li>Screenshots: see <code>docs/DEMO_SCRIPT.md</code> for the shot list.</li>
        </ul>
      </Card>
    </div>
  );
}
