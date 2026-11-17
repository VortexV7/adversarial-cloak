import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, FileText, Image as ImageIcon, Moon, Settings as SettingsIcon, Shield, Sun, UserPlus, Video } from "lucide-react";
import Benchmarks from "./pages/Benchmarks";
import Enrollment from "./pages/Enrollment";
import ImageCloak from "./pages/ImageCloak";
import LiveMonitor from "./pages/LiveMonitor";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import { api, type CloakConfig } from "./lib";
import { Toast } from "./components/ui";

type Page = "live" | "cloak" | "enroll" | "benchmarks" | "reports" | "settings";

const NAV: { id: Page; label: string; icon: typeof Video }[] = [
  { id: "live", label: "Live Monitor", icon: Video },
  { id: "cloak", label: "Image Cloak", icon: ImageIcon },
  { id: "enroll", label: "Enrollment", icon: UserPlus },
  { id: "benchmarks", label: "Benchmarks", icon: BarChart3 },
  { id: "reports", label: "Reports", icon: FileText },
  { id: "settings", label: "Settings", icon: SettingsIcon },
];

export default function App() {
  const [page, setPage] = useState<Page>("live");
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("cloak-theme");
    return saved ? saved === "dark" : true;
  });
  const [cfg, setCfg] = useState<CloakConfig | null>(null);
  const [health, setHealth] = useState<{ status: string; device: string; device_mode?: string; using_gpu?: boolean } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("cloak-theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    api<CloakConfig>("/config").then(setCfg).catch(() => setToast("Backend not reachable at localhost:8000 — start it with `make dev`."));
    api<{ status: string; device: string; device_mode?: string; using_gpu?: boolean }>("/health").then(setHealth).catch(() => undefined);
  }, []);

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-ink-950 dark:text-slate-100">
      <header className="sticky top-0 z-40 border-b border-slate-200/60 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-ink-900/80">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-card bg-teal-500 text-white"><Shield size={20} /></span>
            <div>
              <h1 className="text-lg font-extrabold leading-tight tracking-tight">Adversarial Cloak</h1>
              <p className="hidden text-xs text-slate-500 dark:text-slate-400 sm:block">Real-time face privacy filter</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {health?.device_mode && (
              <span role="status" aria-label="Compute device"
                title={health.using_gpu ? "AI models are running on the GPU" : "Running on the CPU — pick GPU or Both in Settings for faster processing"}
                className={`hidden items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold md:flex ${health.using_gpu ? "bg-teal-500/15 text-teal-700 dark:text-teal-300" : "bg-slate-500/15 text-slate-600 dark:text-slate-300"}`}>
                {health.using_gpu ? "⚡ GPU" : "CPU"} · {health.device_mode}
              </span>
            )}
            <span role="status" aria-label="Connection status"
              className={`hidden items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold sm:flex ${health ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400" : "bg-amber-500/15 text-amber-600 dark:text-amber-400"}`}>
              <span className={`h-2 w-2 rounded-full ${health ? "bg-emerald-500" : "bg-amber-500"}`} />
              {health ? "Connected" : "Backend offline"}
            </span>
            <button onClick={() => setDark(!dark)} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
              className="rounded-card border border-slate-300 p-2 dark:border-slate-600">
              {dark ? <Sun size={17} /> : <Moon size={17} />}
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-7xl gap-4 px-4 py-4">
        <nav aria-label="Main navigation" className="hidden w-52 shrink-0 flex-col gap-1 md:flex">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setPage(id)} aria-current={page === id ? "page" : undefined}
              className={`flex items-center gap-2.5 rounded-card px-3 py-2.5 text-left text-sm font-medium transition-colors ${page === id ? "bg-teal-500/15 text-teal-700 dark:text-teal-300" : "text-slate-600 hover:bg-slate-200/60 dark:text-slate-300 dark:hover:bg-slate-800/60"}`}>
              <Icon size={17} /> {label}
            </button>
          ))}
          <p className="mt-4 px-3 text-xs leading-relaxed text-slate-400">
            What it does: hides your face from face-recognition AI — the video looks the same to people, but machines can't match who you are.
          </p>
          <p className="mt-2 px-3 text-xs leading-relaxed text-slate-400">
            Defensive privacy tool. Protects against passive scraping and off-the-shelf recognisers — not adaptive attackers. See Reports for the full threat model.
          </p>
        </nav>

        {/* Mobile nav */}
        <nav aria-label="Main navigation mobile" className="fixed bottom-0 left-0 right-0 z-40 flex justify-around border-t border-slate-200 bg-white/95 py-2 backdrop-blur md:hidden dark:border-slate-800 dark:bg-ink-900/95">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setPage(id)} aria-current={page === id ? "page" : undefined}
              className={`flex flex-col items-center gap-0.5 text-[10px] font-medium ${page === id ? "text-teal-600 dark:text-teal-400" : "text-slate-500"}`}>
              <Icon size={19} /> {label.split(" ")[0]}
            </button>
          ))}
        </nav>

        <main className="min-w-0 flex-1 pb-16 md:pb-0">
          <AnimatePresence mode="wait">
            <motion.div key={page} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}>
              {page === "live" && <LiveMonitor cfg={cfg} setCfg={setCfg} onToast={setToast} />}
              {page === "cloak" && <ImageCloak onToast={setToast} />}
              {page === "enroll" && <Enrollment onToast={setToast} />}
              {page === "benchmarks" && <Benchmarks onToast={setToast} />}
              {page === "reports" && <Reports onToast={setToast} />}
              {page === "settings" && <Settings cfg={cfg} setCfg={setCfg} dark={dark} setDark={setDark} onToast={setToast} />}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      {toast && <Toast message={toast} onClose={() => setToast(null)} />}
    </div>
  );
}
