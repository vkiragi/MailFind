import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api, type AccountSummary, type SyncStatus } from "@/lib/api";

interface Props {
  account: AccountSummary;
  onChange: () => void;
}

export default function SyncPanel({ account, onChange }: Props) {
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [account.id]);

  const runSync = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.syncNow(account.id, false);
      await refresh();
      onChange();
    } catch (err) {
      setError(String(err));
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

        <div className="flex gap-2">
          <Button onClick={runSync} disabled={busy || status?.is_running}>
            {status?.is_running ? "Syncing…" : "Sync Now"}
          </Button>
          <Button variant="outline" onClick={ingestFixture} disabled={busy}>
            Import .eml file
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
