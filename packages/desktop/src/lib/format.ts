/** Compact relative time, e.g. "just now", "3h ago", "2d ago", "5mo ago". */
export function relativeTime(dateStr: string): string {
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return "";
  const sec = Math.floor((Date.now() - d.getTime()) / 1000);
  if (sec < 45) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day}d ago`;
  if (day < 30) return `${Math.floor(day / 7)}w ago`;
  if (day < 365) return `${Math.floor(day / 30)}mo ago`;
  return `${Math.floor(day / 365)}y ago`;
}

// Tinted avatar swatches — literal class strings so Tailwind's JIT keeps them,
// and each pair is legible in both light and dark.
const AVATAR_COLORS = [
  "bg-blue-500/15 text-blue-600 dark:text-blue-300",
  "bg-violet-500/15 text-violet-600 dark:text-violet-300",
  "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
  "bg-amber-500/15 text-amber-700 dark:text-amber-300",
  "bg-rose-500/15 text-rose-600 dark:text-rose-300",
  "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300",
  "bg-indigo-500/15 text-indigo-600 dark:text-indigo-300",
  "bg-teal-500/15 text-teal-700 dark:text-teal-300",
];

function deriveInitials(s: string): string {
  const base = (s.replace(/<[^>]*>/, "").trim() || s).trim();
  const words = base.split(/\s+/).filter(Boolean);
  if (words.length >= 2 && /[a-z]/i.test(words[0]) && /[a-z]/i.test(words[1])) {
    return (words[0][0] + words[1][0]).toUpperCase();
  }
  const local = (words[0] || base).split("@")[0];
  return (local.slice(0, 2) || "?").toUpperCase();
}

/** Deterministic initials + tinted color classes for a sender name or email. */
export function avatar(input: string): { initials: string; colorClass: string } {
  const clean = (input || "").trim();
  let hash = 0;
  for (let i = 0; i < clean.length; i++) {
    hash = (hash * 31 + clean.charCodeAt(i)) | 0;
  }
  const idx = Math.abs(hash) % AVATAR_COLORS.length;
  return { initials: deriveInitials(clean), colorClass: AVATAR_COLORS[idx] };
}
