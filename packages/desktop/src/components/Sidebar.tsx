import { Search, Sparkles, Settings, HelpCircle } from "lucide-react";
import ModelStatusBadge from "@/components/ModelStatusBadge";
import { cn } from "@/lib/utils";

export type Tab = "search" | "chat" | "settings" | "help";

// Ask listed first: it's the fuller experience (Search's own retrieval, plus
// a synthesized, cited answer) and stays clickable even when gated off on
// weaker hardware — clicking it there just explains why and how to enable
// it, which is a better default than burying the capability behind Search.
const NAV: { id: Tab; label: string; icon: typeof Search }[] = [
  { id: "chat", label: "Ask", icon: Sparkles },
  { id: "search", label: "Search", icon: Search },
  { id: "settings", label: "Accounts", icon: Settings },
  { id: "help", label: "Help", icon: HelpCircle },
];

function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 1024 1024" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="sbLogo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#0ea5e9" />
          <stop offset="1" stopColor="#2563eb" />
        </linearGradient>
      </defs>
      <rect width="1024" height="1024" rx="224" fill="url(#sbLogo)" />
      <rect
        x="248"
        y="304"
        width="440"
        height="312"
        rx="48"
        fill="none"
        stroke="#fff"
        strokeWidth="38"
      />
      <path
        d="M266 350 L468 496 L670 350"
        fill="none"
        stroke="#fff"
        strokeWidth="38"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="690" cy="690" r="118" fill="url(#sbLogo)" stroke="#fff" strokeWidth="38" />
      <line x1="774" y1="774" x2="858" y2="858" stroke="#fff" strokeWidth="52" strokeLinecap="round" />
    </svg>
  );
}

export default function Sidebar({
  tab,
  onTab,
  statusKey,
}: {
  tab: Tab;
  onTab: (t: Tab) => void;
  statusKey?: number;
}) {
  return (
    <aside className="flex w-[228px] shrink-0 flex-col border-r border-border bg-card">
      <div className="flex items-center gap-2.5 px-5 pb-5 pt-6">
        <LogoMark className="size-8 rounded-[9px] shadow-soft" />
        <span className="font-display text-lg font-semibold tracking-tight text-foreground">
          MailFind
        </span>
      </div>

      <nav className="flex flex-col gap-1 px-3">
        {NAV.map(({ id, label, icon: Icon }) => {
          const active = tab === id;
          return (
            <button
              key={id}
              onClick={() => onTab(id)}
              className={cn(
                "group relative flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-secondary/70 hover:text-foreground",
              )}
            >
              {active && (
                <span className="absolute left-0 top-1/2 h-5 w-1 -translate-y-1/2 rounded-full bg-brand-gradient" />
              )}
              <Icon className="size-[18px]" />
              {label}
            </button>
          );
        })}
      </nav>

      <div className="mt-auto p-3">
        <ModelStatusBadge key={statusKey} />
      </div>
    </aside>
  );
}
