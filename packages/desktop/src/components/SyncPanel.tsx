import { useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  api,
  IMPORT_PROGRESS_EVENT,
  SYNC_PROGRESS_EVENT,
  type AccountSummary,
  type ImportProgress,
  type ImportResult,
  type SyncProgress,
  type SyncStatus,
} from "@/lib/api";

interface Props {
  account: AccountSummary;
  onChange: () => void;
}

const COOLDOWN_MS = 10 * 60 * 1000;
const AUTO_SYNC_INTERVAL_MS = 5 * 60 * 1000;
// Full IMAP backfill window for "Sync more history" (also used on first run).
const HISTORY_SYNC_WINDOW = "three_months" as const;

function isRateLimited(err: string): boolean {
  const lower = err.toLowerCase();
  return (
    lower.includes("rate") ||
    lower.includes("throttl") ||
    lower.includes("unavailable")
  );
}

// Backend returns this when a sync is already in flight for the account.
// Usually a race between the auto-sync timer and a manual click — not really
// an error worth showing the user, just refresh status and move on.
function isAlreadyRunning(err: string): boolean {
  return err.toLowerCase().includes("sync is already running");
}

export default function SyncPanel({ account, onChange }: Props) {
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<SyncProgress | null>(null);
  const [importProgress, setImportProgress] = useState<ImportProgress | null>(
    null,
  );
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  // Guard against re-firing the first-run backfill on every status refresh.
  // Reset by account.id so switching accounts re-evaluates.
  const firstRunDone = useRef(false);
  // Wall-clock timestamp until which Sync is locally disabled. Backed up in
  // localStorage so a reload doesn't let the user bypass it.
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
    firstRunDone.current = false;
    refresh();
    api
      .syncCooldownUntil(account.id)
      .then((until) => {
        if (until > Date.now()) {
          setCooldownUntil(until);
          localStorage.setItem(`mf:cooldown:${account.id}`, String(until));
        } else {
          localStorage.removeItem(`mf:cooldown:${account.id}`);
          setCooldownUntil(0);
        }
      })
      .catch(() => {});
    const t = setInterval(refresh, 4000);
    let unlistenSync: UnlistenFn | undefined;
    let unlistenImport: UnlistenFn | undefined;
    listen<SyncProgress>(SYNC_PROGRESS_EVENT, (e) => {
      if (e.payload.account_id !== account.id) return;
      setProgress(e.payload);
      if (e.payload.phase === "done") {
        setTimeout(() => setProgress(null), 2000);
      }
    }).then((fn) => {
      unlistenSync = fn;
    });
    listen<ImportProgress>(IMPORT_PROGRESS_EVENT, (e) => {
      if (e.payload.account_id !== account.id) return;
      setImportProgress(e.payload);
      if (e.payload.done) {
        setTimeout(() => setImportProgress(null), 2000);
      }
    }).then((fn) => {
      unlistenImport = fn;
    });
    return () => {
      clearInterval(t);
      unlistenSync?.();
      unlistenImport?.();
    };
  }, [account.id]);

  // First-run backfill: when this account has no messages yet, populate from
  // Apple Mail (instant, local) and then run a full IMAP fetch for the
  // default window so the watermark is established and anything iCloud has
  // that Apple Mail missed comes through. Dedup on Message-ID keeps this
  // idempotent if anything overlaps.
  useEffect(() => {
    if (firstRunDone.current) return;
    if (!status) return;
    if (status.total_messages > 0) {
      firstRunDone.current = true;
      return;
    }
    firstRunDone.current = true;
    (async () => {
      setImporting(true);
      setError(null);
      setImportError(null);
      try {
        const scan = await api.scanAppleMail();
        if (scan.mail_dir && scan.message_count > 0) {
          try {
            const result = await api.importAppleMail(account.id);
            setImportResult(result);
            await refresh();
            onChange();
          } catch (err) {
            setImportError(String(err));
          }
        }
        // Always follow with an IMAP backfill to set up the UID watermark
        // and catch server-side mail Apple Mail didn't have.
        await api.syncNow(account.id, true, HISTORY_SYNC_WINDOW);
        await refresh();
        onChange();
      } catch (err) {
        setError(String(err));
      } finally {
        setImporting(false);
      }
    })();
  }, [status, account.id, onChange]);

  // Background auto-sync: incremental fetch every N minutes and whenever the
  // window regains focus. Skips when busy, already syncing, or in cooldown.
  useEffect(() => {
    const tick = async () => {
      if (busy || importing) return;
      if (status?.is_running) return;
      if (cooldownUntil > Date.now()) return;
      // Don't auto-sync until the first-run backfill has done its job.
      if (!status || status.total_messages === 0) return;
      try {
        await api.syncNow(account.id, false);
        await refresh();
        onChange();
      } catch {
        // Auto-sync failures are silent — a manual Sync will surface them.
      }
    };
    const t = setInterval(tick, AUTO_SYNC_INTERVAL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") tick();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      clearInterval(t);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [account.id, busy, importing, status, cooldownUntil, onChange]);

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
      await api.syncNow(account.id, false);
      await refresh();
      onChange();
    } catch (err) {
      const msg = String(err);
      setProgress(null);
      if (isAlreadyRunning(msg)) {
        // Auto-sync raced us; the other one will produce results. Just refresh.
        await refresh();
      } else {
        setError(msg);
        if (isRateLimited(msg)) {
          const until = Date.now() + COOLDOWN_MS;
          setCooldownUntil(until);
          localStorage.setItem(`mf:cooldown:${account.id}`, String(until));
        }
      }
    } finally {
      setBusy(false);
    }
  };

  const runHistorySync = async () => {
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
      await api.syncNow(account.id, true, HISTORY_SYNC_WINDOW);
      await refresh();
      onChange();
    } catch (err) {
      const msg = String(err);
      setProgress(null);
      if (isAlreadyRunning(msg)) {
        // Auto-sync raced us; the other one will produce results. Just refresh.
        await refresh();
      } else {
        setError(msg);
        if (isRateLimited(msg)) {
          const until = Date.now() + COOLDOWN_MS;
          setCooldownUntil(until);
          localStorage.setItem(`mf:cooldown:${account.id}`, String(until));
        }
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

  const importPct =
    importProgress && importProgress.total > 0
      ? Math.min(100, (importProgress.current / importProgress.total) * 100)
      : 0;

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
        {(importing || importProgress) && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Setting up your mailbox…</span>
              <span>
                {importProgress
                  ? `${importProgress.current.toLocaleString()} / ${importProgress.total.toLocaleString()}`
                  : ""}
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={`h-full rounded-full bg-primary transition-all ${
                  importProgress ? "" : "animate-pulse w-1/3"
                }`}
                style={importProgress ? { width: `${importPct}%` } : undefined}
              />
            </div>
          </div>
        )}
        {importResult && !importing && (
          <div className="rounded-md border border-green-500/40 bg-green-500/10 p-2 text-xs text-green-700 dark:text-green-400">
            Loaded {importResult.imported.toLocaleString()} message
            {importResult.imported === 1 ? "" : "s"} from Apple Mail.
          </div>
        )}
        {importError && (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-400">
            Apple Mail backfill skipped: {importError}
          </div>
        )}
        {progress && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{phaseLabel(progress.phase)}</span>
              <span>{progressLabel(progress)}</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={`h-full rounded-full transition-all ${
                  progress.phase === "failed" ? "bg-destructive" : "bg-primary"
                } ${progress.total == null ? "animate-pulse w-1/3" : ""}`}
                style={
                  progress.total != null && progress.total > 0
                    ? {
                        width: `${Math.min(100, (progress.current / progress.total) * 100)}%`,
                      }
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
          <Button
            onClick={runSync}
            disabled={
              busy || importing || status?.is_running || cooldownUntil > Date.now()
            }
          >
            {cooldownUntil > Date.now()
              ? `Wait ${formatRemaining(cooldownUntil - Date.now())}`
              : status?.is_running
                ? "Syncing…"
                : "Sync"}
          </Button>
          <Button
            variant="outline"
            onClick={runHistorySync}
            disabled={
              busy || importing || status?.is_running || cooldownUntil > Date.now()
            }
            title="Fetches up to 90 days from iCloud"
          >
            Sync more history
          </Button>
          <Button variant="outline" onClick={ingestFixture} disabled={busy}>
            Import .eml file
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Sync fetches new mail since your last sync. Sync more history re-fetches
          up to 90 days from iCloud — use this if you haven&apos;t opened the app
          in a while.
        </p>
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
