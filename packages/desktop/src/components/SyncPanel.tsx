import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  api,
  SYNC_PROGRESS_EVENT,
  type AccountSummary,
  type SyncProgress,
  type SyncStatus,
  type SyncWindow,
} from "@/lib/api";

interface Props {
  account: AccountSummary;
  onChange: () => void;
}

const WINDOW_OPTIONS: Array<{
  value: SyncWindow | "six_months" | "one_year";
  label: string;
  disabled?: boolean;
}> = [
  { value: "day", label: "Last day" },
  { value: "week", label: "Last 7 days" },
  { value: "two_weeks", label: "Last 2 weeks" },
  { value: "month", label: "Last month" },
  { value: "three_months", label: "Last 3 months" },
  { value: "six_months", label: "Last 6 months (slow — coming soon)", disabled: true },
  { value: "one_year", label: "Last year (slow — coming soon)", disabled: true },
];

const COOLDOWN_MS = 10 * 60 * 1000;

function isRateLimited(err: string): boolean {
  const lower = err.toLowerCase();
  return (
    lower.includes("rate") ||
    lower.includes("throttl") ||
    lower.includes("unavailable")
  );
}

export default function SyncPanel({ account, onChange }: Props) {
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [window, setWindow] = useState<SyncWindow>("month");
  const [progress, setProgress] = useState<SyncProgress | null>(null);
  // Wall-clock timestamp until which Sync Now is locally disabled. Backed up
  // in localStorage so a reload doesn't let the user bypass it.
  const [cooldownUntil, setCooldownUntil] = useState<number>(() => {
    const stored = localStorage.getItem(`mf:cooldown:${account.id}`);
    const n = stored ? Number(stored) : 0;
    return Number.isFinite(n) && n > Date.now() ? n : 0;
  });
  const [, forceTick] = useState(0);

  const refresh = async () => {
    try {
      const s = await api.syncStatus(account.id);
      setStatus(s);
    } catch (err) {
      setError(String(err));
    }
  };

  useEffect(() => {
    refresh();
    // Reconcile cooldown with the backend, which is the source of truth and
    // persists across app restarts. The localStorage fallback only matters
    // when the backend hasn't loaded yet.
    api
      .syncCooldownUntil(account.id)
      .then((until) => {
        if (until > Date.now()) {
          setCooldownUntil(until);
          localStorage.setItem(`mf:cooldown:${account.id}`, String(until));
        } else {
          // Backend says no cooldown — clear any stale localStorage entry.
          localStorage.removeItem(`mf:cooldown:${account.id}`);
          setCooldownUntil(0);
        }
      })
      .catch(() => {
        // Backend unreachable; keep whatever localStorage said.
      });
    const t = setInterval(refresh, 4000);
    let unlisten: UnlistenFn | undefined;
    listen<SyncProgress>(SYNC_PROGRESS_EVENT, (e) => {
      if (e.payload.account_id !== account.id) return;
      setProgress(e.payload);
      // Auto-clear only on success. Failures stay visible so the user can
      // see what happened; clicking Sync Now again will overwrite it.
      if (e.payload.phase === "done") {
        setTimeout(() => setProgress(null), 2000);
      }
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      clearInterval(t);
      unlisten?.();
    };
  }, [account.id]);

  // Re-render every second while in cooldown so the countdown ticks down.
  useEffect(() => {
    if (cooldownUntil <= Date.now()) return;
    const t = setInterval(() => forceTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [cooldownUntil]);

  const runSync = async () => {
    if (cooldownUntil > Date.now()) return;
    setBusy(true);
    setError(null);
    setProgress({
      account_id: account.id,
      phase: "connecting",
      total: null,
      current: 0,
      message: null,
    });
    try {
      await api.syncNow(account.id, false, window);
      await refresh();
      onChange();
    } catch (err) {
      const msg = String(err);
      setError(msg);
      setProgress(null);
      if (isRateLimited(msg)) {
        const until = Date.now() + COOLDOWN_MS;
        setCooldownUntil(until);
        localStorage.setItem(`mf:cooldown:${account.id}`, String(until));
      }
    } finally {
      setBusy(false);
    }
  };

  const ingestFixture = async () => {
    setBusy(true);
    setError(null);
    try {
      const path = await open({
        multiple: false,
        directory: false,
        filters: [
          { name: "Email", extensions: ["eml", "mbox", "txt"] },
          { name: "All", extensions: ["*"] },
        ],
      });
      if (!path) {
        setBusy(false);
        return;
      }
      const result = await api.ingestFixture({
        account_id: account.id,
        path: path as string,
      });
      await refresh();
      onChange();
      if (result.errors.length > 0) {
        setError(result.errors.join("; "));
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{account.email}</CardTitle>
        <p className="text-xs text-muted-foreground">
          {account.imap_host} • added{" "}
          {new Date(account.created_at).toLocaleDateString()}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {progress && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{phaseLabel(progress.phase)}</span>
              <span>{progressLabel(progress)}</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={`h-full rounded-full transition-all ${
                  progress.phase === "failed"
                    ? "bg-destructive"
                    : "bg-primary"
                } ${
                  progress.total == null ? "animate-pulse w-1/3" : ""
                }`}
                style={
                  progress.total != null && progress.total > 0
                    ? { width: `${Math.min(100, (progress.current / progress.total) * 100)}%` }
                    : undefined
                }
              />
            </div>
          </div>
        )}
        <div className="grid grid-cols-3 gap-3 text-xs">
          <div>
            <div className="text-muted-foreground">Messages</div>
            <div className="text-base font-semibold">
              {status?.total_messages ?? "—"}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">Embedded</div>
            <div className="text-base font-semibold">
              {status?.embedded_messages ?? "—"}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">Last sync</div>
            <div className="text-base font-semibold">
              {status?.last_sync
                ? new Date(status.last_sync).toLocaleString()
                : "Never"}
            </div>
          </div>
        </div>

        {status?.last_error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
            {status.last_error}
          </div>
        )}
        {error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
            {error}
          </div>
        )}
        {cooldownUntil > Date.now() && (
          <div className="rounded-md border border-amber-500/50 bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-400">
            Mail server is throttling this account. Sync paused for{" "}
            {formatRemaining(cooldownUntil - Date.now())} to let it recover.
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <select
            className="rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            value={window}
            onChange={(e) => setWindow(e.target.value as SyncWindow)}
            disabled={busy || status?.is_running || cooldownUntil > Date.now()}
          >
            {WINDOW_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value} disabled={opt.disabled}>
                {opt.label}
              </option>
            ))}
          </select>
          <Button
            onClick={runSync}
            disabled={busy || status?.is_running || cooldownUntil > Date.now()}
          >
            {cooldownUntil > Date.now()
              ? `Wait ${formatRemaining(cooldownUntil - Date.now())}`
              : status?.is_running
                ? "Syncing…"
                : "Sync Now"}
          </Button>
          <Button variant="outline" onClick={ingestFixture} disabled={busy}>
            Import .eml file
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function phaseLabel(phase: SyncProgress["phase"]): string {
  switch (phase) {
    case "connecting":
      return "Connecting…";
    case "searching":
      return "Searching mailbox…";
    case "fetching":
      return "Fetching messages";
    case "storing":
      return "Storing messages";
    case "done":
      return "Done";
    case "failed":
      return "Failed";
  }
}

function progressLabel(p: SyncProgress): string {
  if (p.total != null && p.total > 0) {
    return `${p.current} / ${p.total}`;
  }
  return p.current > 0 ? `${p.current}` : "";
}

function formatRemaining(ms: number): string {
  const total = Math.max(0, Math.ceil(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  if (m > 0) return `${m}m ${s.toString().padStart(2, "0")}s`;
  return `${s}s`;
}
