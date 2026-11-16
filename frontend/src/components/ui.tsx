import { motion } from "framer-motion";
import type { ReactNode } from "react";

export function Card({ children, className = "", label }: { children: ReactNode; className?: string; label?: string }) {
  return (
    <section aria-label={label} className={`glass rounded-card shadow-sm p-4 sm:p-5 ${className}`}>
      {children}
    </section>
  );
}

export function CardTitle({ children }: { children: ReactNode }) {
  return <h2 className="text-sm font-semibold tracking-wide text-slate-500 dark:text-slate-400 uppercase">{children}</h2>;
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      role="switch" aria-checked={checked} aria-label={label}
      onClick={() => onChange(!checked)}
      className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${checked ? "bg-teal-500" : "bg-slate-300 dark:bg-slate-600"}`}
    >
      <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${checked ? "left-[22px]" : "left-0.5"}`} />
    </button>
  );
}

export function Slider({ label, value, min, max, step, onChange, format }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <label className="block">
      <span className="flex justify-between text-sm">
        <span className="font-medium text-slate-700 dark:text-slate-200">{label}</span>
        <span className="tabular-nums text-teal-600 dark:text-teal-400 font-semibold">{format ? format(value) : value}</span>
      </span>
      <input type="range" aria-label={label} min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} className="mt-1 w-full" />
    </label>
  );
}

export function Gauge({ value, label, sub }: { value: number; label: string; sub: string }) {
  const pct = Math.max(0, Math.min(1, value));
  const color = pct > 0.6 ? "#f43f5e" : pct > 0.25 ? "#f59e0b" : "#10b981";
  const r = 52, circ = 2 * Math.PI * r;
  return (
    <div className="flex flex-col items-center gap-1" role="img" aria-label={`${label}: ${Math.round(pct * 100)} percent`}>
      <svg width="132" height="132" viewBox="0 0 132 132" className=" -rotate-90">
        <circle cx="66" cy="66" r={r} fill="none" strokeWidth="10" className="stroke-slate-200 dark:stroke-slate-700" />
        <motion.circle cx="66" cy="66" r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={circ} animate={{ strokeDashoffset: circ * (1 - pct) }} transition={{ duration: 0.5 }} />
      </svg>
      <div className="-mt-[86px] text-2xl font-bold tabular-nums">{Math.round(pct * 100)}%</div>
      <div className="mt-[52px] text-sm font-semibold">{label}</div>
      <div className="text-xs text-slate-500 dark:text-slate-400">{sub}</div>
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden className={`animate-pulse rounded-card bg-slate-200/70 dark:bg-slate-700/50 ${className}`} />;
}

export function Toast({ message, onClose }: { message: string; onClose: () => void }) {
  return (
    <motion.div role="alert" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
      className="fixed bottom-4 right-4 z-50 flex items-center gap-3 rounded-card bg-rose-600 px-4 py-3 text-sm font-medium text-white shadow-lg">
      {message}
      <button onClick={onClose} aria-label="Dismiss error" className="font-bold">✕</button>
    </motion.div>
  );
}
