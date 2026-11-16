import { useEffect, useRef, useState } from "react";
import { Upload, UserCheck } from "lucide-react";
import { api } from "../lib";
import { Card, CardTitle, Skeleton } from "../components/ui";

export default function Enrollment({ onToast }: { onToast: (m: string) => void }) {
  const [name, setName] = useState("Ved Sharanagate");
  const [preview, setPreview] = useState<string | null>(null);
  const [enrolled, setEnrolled] = useState<{ name: string; enrolled_at?: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api<{ name: string | null; enrolled: boolean; enrolled_at?: number }>("/enroll")
      .then((r) => { if (r.enrolled && r.name) setEnrolled({ name: r.name, enrolled_at: r.enrolled_at }); })
      .catch(() => { /* backend not up yet */ });
  }, []);

  const onFile = async (file: File) => {
    setPreview(URL.createObjectURL(file));
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`/api/enroll?name=${encodeURIComponent(name)}`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      setEnrolled({ name: data.name, enrolled_at: data.enrolled_at });
    } catch {
      onToast("Enrolment failed — is the backend running?");
    } finally { setBusy(false); }
  };

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card label="Enroll a reference face">
        <CardTitle>Enroll a reference face</CardTitle>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          The reference embedding is stored locally (backend <code>enrolled.json</code>) and used as the
          identity the cloak must move away from. Only the 512-d embedding is kept — never the photo.
        </p>
        <label className="mt-4 block text-sm font-medium">Display name
          <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Display name"
            className="mt-1 w-full rounded-card border border-slate-300 bg-transparent px-3 py-2 dark:border-slate-600" />
        </label>
        <input ref={inputRef} type="file" accept="image/*" className="hidden"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        <button onClick={() => inputRef.current?.click()} disabled={busy}
          className="mt-4 flex items-center gap-2 rounded-card bg-teal-500 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
          <Upload size={16} /> {busy ? "Enrolling…" : "Upload a face photo"}
        </button>
        {busy && <Skeleton className="mt-4 h-40 w-full" />}
        {preview && !busy && <img src={preview} alt="Uploaded reference face preview" className="mt-4 max-h-64 rounded-card object-cover" />}
      </Card>

      <Card label="Identity card">
        <CardTitle>Identity card</CardTitle>
        {enrolled ? (
          <div className="mt-4 flex items-start gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-teal-500/15 text-teal-600 dark:text-teal-400"><UserCheck /></div>
            <div>
              <p className="text-lg font-bold">{enrolled.name}</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">Reference embedding stored · 512-d · ArcFace evaluator</p>
              {enrolled.enrolled_at && <p className="text-xs text-slate-400">Enrolled {new Date(enrolled.enrolled_at * 1000).toLocaleString()}</p>}
              <button onClick={async () => { await api("/enroll", { method: "DELETE" }); setEnrolled(null); setPreview(null); }}
                className="mt-3 rounded-card border border-rose-300 px-3 py-1.5 text-sm font-semibold text-rose-600 dark:border-rose-700">Remove enrolment</button>
            </div>
          </div>
        ) : (
          <div className="mt-4 rounded-card border-2 border-dashed border-slate-300 p-6 text-center text-sm text-slate-500 dark:border-slate-600 dark:text-slate-400">
            No identity enrolled yet. Upload a face photo to create an identity card — the Live Monitor will then show confidence against <i>your</i> face instead of the frame itself.
          </div>
        )}
      </Card>
    </div>
  );
}
