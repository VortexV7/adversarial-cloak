import { useRef, useState } from "react";
import { Download, ImagePlus, Loader2, ShieldCheck, ShieldAlert } from "lucide-react";
import { API, qualityBadge, type CloakImageResult } from "../lib";
import { Card, CardTitle, InfoButton, Skeleton } from "../components/ui";

export default function ImageCloak({ onToast }: { onToast: (m: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<CloakImageResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pick = (f: File | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) { onToast("Please choose an image file (JPEG, PNG or WebP)."); return; }
    setFile(f);
    setResult(null);
    const reader = new FileReader();
    reader.onload = () => setPreview(String(reader.result));
    reader.readAsDataURL(f);
  };

  const cloak = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(API("/cloak-image"), { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `Cloaking failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Cloaking failed.");
    } finally {
      setBusy(false);
    }
  };

  const download = () => {
    if (!result) return;
    const a = document.createElement("a");
    a.href = `data:image/jpeg;base64,${result.image}`;
    a.download = "cloaked.jpg";
    a.click();
  };

  const img = (b64?: string) => b64 ? `data:image/jpeg;base64,${b64}` : undefined;

  return (
    <div className="space-y-4">
      <Card label="Image cloak">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle>Image cloak</CardTitle>
          <InfoButton label="Image cloak" text="Upload a photo and the cloak is added to every face in it. The photo looks the same to you, but face-recognition AI can no longer match the people in it." />
        </div>
        <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
          Add a photo of someone and the cloak is baked into the image itself — share it anywhere and
          face-recognition systems can't identify the people in it, while the photo still looks unchanged.
        </p>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); pick(e.dataTransfer.files[0]); }}
          onClick={() => inputRef.current?.click()}
          className={`mt-4 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-card border-2 border-dashed px-4 py-10 text-center transition-colors ${dragOver ? "border-teal-500 bg-teal-500/10" : "border-slate-300 dark:border-slate-600"}`}>
          <ImagePlus size={32} className="text-slate-400" />
          <p className="font-medium">{file ? file.name : "Drop a photo here, or click to choose one"}</p>
          <p className="text-sm text-slate-500 dark:text-slate-400">JPEG, PNG or WebP · processed on your backend, never uploaded anywhere else</p>
          <input ref={inputRef} type="file" accept="image/*" className="hidden"
            onChange={(e) => pick(e.target.files?.[0])} />
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <button onClick={cloak} disabled={!file || busy}
            className="flex items-center gap-1.5 rounded-card bg-teal-500 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
            {busy ? <Loader2 size={16} className="animate-spin" /> : <ShieldCheck size={16} />}
            {busy ? "Cloaking…" : "Cloak this photo"}
          </button>
          {result && (
            <button onClick={download} className="flex items-center gap-1.5 rounded-card border border-slate-300 px-4 py-2 text-sm font-semibold dark:border-slate-600">
              <Download size={16} /> Download cloaked photo
            </button>
          )}
        </div>
      </Card>

      {busy && <Skeleton className="h-64 w-full" />}

      {result && (
        <Card label="Cloaked photo">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle>Result</CardTitle>
            {result.attack_success
              ? <span className="flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-600 dark:text-emerald-400"><ShieldCheck size={14} /> Identity hidden</span>
              : <span className="flex items-center gap-1.5 rounded-full bg-amber-500/15 px-3 py-1 text-xs font-semibold text-amber-600 dark:text-amber-400"><ShieldAlert size={14} /> Similarity still above the match threshold</span>}
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <figure className="relative">
              <img src={img(result.original)} alt="Original photo" className="w-full rounded-card object-cover" />
              <figcaption className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs font-semibold text-white">Original</figcaption>
            </figure>
            <figure className="relative">
              <img src={img(result.image)} alt="Cloaked photo" className="w-full rounded-card object-cover" />
              <figcaption className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs font-semibold text-white">Cloaked</figcaption>
            </figure>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2 text-center sm:grid-cols-4">
            <div className="rounded-card bg-slate-100/70 p-3 dark:bg-slate-800/60">
              <div className="text-xs text-slate-500 dark:text-slate-400">Faces cloaked</div>
              <div className="text-xl font-bold tabular-nums">{result.faces}</div>
            </div>
            <div className="rounded-card bg-slate-100/70 p-3 dark:bg-slate-800/60">
              <div className="text-xs text-slate-500 dark:text-slate-400">Recognition similarity</div>
              <div className="text-xl font-bold tabular-nums">{result.similarity_before.toFixed(2)} → {result.similarity_after.toFixed(2)}</div>
            </div>
            <div className="rounded-card bg-slate-100/70 p-3 dark:bg-slate-800/60">
              <div className="text-xs text-slate-500 dark:text-slate-400">Match confidence</div>
              <div className="text-xl font-bold tabular-nums">{Math.round(result.confidence_before * 100)}% → {Math.round(result.confidence_after * 100)}%</div>
            </div>
            <div className="rounded-card bg-slate-100/70 p-3 dark:bg-slate-800/60">
              <div className="text-xs text-slate-500 dark:text-slate-400">Looks identical (SSIM)</div>
              <div className="text-xl font-bold tabular-nums">{result.ssim.toFixed(3)}</div>
              <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${qualityBadge(result.ssim, 0.95, 0.85)}`}>near-identical</span>
            </div>
          </div>
          <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
            Cloaked in {result.latency_ms.toFixed(0)} ms on {result.device_mode} ({result.device}) · PSNR {result.psnr.toFixed(1)} dB.
            The downloaded photo carries the cloak permanently — it works in any app that reads the image.
          </p>
        </Card>
      )}

      {preview && !result && !busy && (
        <Card label="Photo preview">
          <CardTitle>Your photo</CardTitle>
          <img src={preview} alt="Uploaded photo preview" className="mt-3 max-h-96 w-auto rounded-card" />
        </Card>
      )}
    </div>
  );
}
