import { useEffect, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { AlertTriangle, Check, Cpu, Download, Loader2, RefreshCw, X } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  api,
  MODEL_PULL_EVENT,
  type ModelList,
  type ModelOption,
  type ModelPullProgress,
} from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  /** Notify parent so it can refresh the header model badge after a change. */
  onChange?: () => void;
}

interface PullState {
  completed: number;
  total: number;
  status: string;
}

function gb(n: number): string {
  return `${Math.round(n)} GB`;
}

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}

export default function ModelPicker({ onChange }: Props) {
  const [data, setData] = useState<ModelList | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pulls, setPulls] = useState<Record<string, PullState>>({});
  const [customModel, setCustomModel] = useState("");

  const load = async () => {
    try {
      const d = await api.listModels();
      setData(d);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  // Stream pull progress from the backend.
  useEffect(() => {
    let un: UnlistenFn | undefined;
    listen<ModelPullProgress>(MODEL_PULL_EVENT, (e) => {
      const p = e.payload;
      setPulls((prev) => {
        const next = { ...prev };
        if (p.done || p.error) delete next[p.model];
        else next[p.model] = { completed: p.completed, total: p.total, status: p.status };
        return next;
      });
      if (p.done) {
        load();
        onChange?.();
      }
      if (p.error && p.error !== "Cancelled") {
        setError(`Couldn't install ${p.model}: ${p.error}`);
      }
    })
      .then((fn) => (un = fn))
      .catch(() => {});
    return () => un?.();
  }, []);

  const choose = async (model: string) => {
    setSaving(model);
    setError(null);
    try {
      await api.setChatModel(model);
      await load();
      onChange?.();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(null);
    }
  };

  const startPull = (model: string) => {
    const m = model.trim();
    if (!m || pulls[m]) return;
    setError(null);
    setPulls((prev) => ({ ...prev, [m]: { completed: 0, total: 0, status: "Starting…" } }));
    api.pullModel(m).catch((e) => {
      setPulls((prev) => {
        const next = { ...prev };
        delete next[m];
        return next;
      });
      setError(String(e));
    });
  };

  const stopPull = (model: string) => {
    api.cancelPull(model).catch(() => {});
  };

  const optionModels = new Set((data?.options ?? []).map((o) => o.model));
  const customPulls = Object.keys(pulls).filter((m) => !optionModels.has(m));

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Cpu className="size-4 text-primary" />
            Ask model
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={load}
            disabled={loading || saving !== null}
            title="Re-scan installed models"
          >
            <RefreshCw className={cn("size-4", loading && "animate-spin")} />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Search runs on any Mac. The Ask tab uses a local chat model — MailFind
          picks the best one your memory can run.
          {data && (
            <>
              {" "}
              Detected {gb(data.total_ram_gb)} RAM (~{gb(data.budget_gb)} free for
              a model).
            </>
          )}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && !data && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Checking your models…
          </div>
        )}

        {data && !data.ollama_reachable && (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-400">
            Ollama isn&apos;t reachable. Start it to enable the Ask tab; search
            still works without it.
          </div>
        )}

        {data && data.ollama_reachable && data.auto_pick_state === "search_only" && (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-400">
            This Mac&apos;s memory is below the threshold for a reliable Ask
            model, so Ask is off — search still works. You can still install a
            small model below.
          </div>
        )}

        {data && (
          <div className="space-y-2">
            {data.options.map((opt) => (
              <ModelRow
                key={opt.model}
                opt={opt}
                recommended={data.auto_pick_model === opt.model}
                saving={saving === opt.model}
                disabled={saving !== null}
                pull={pulls[opt.model]}
                onSelect={() => choose(opt.model)}
                onInstall={() => startPull(opt.model)}
                onCancelInstall={() => stopPull(opt.model)}
              />
            ))}
          </div>
        )}

        {data && data.other_installed.length > 0 && (
          <div className="space-y-2">
            <p className="pt-1 text-xs font-medium text-muted-foreground">
              Other installed models
            </p>
            {data.other_installed.map((model) => (
              <OtherModelRow
                key={model}
                model={model}
                current={data.current_model === model}
                saving={saving === model}
                disabled={saving !== null}
                onSelect={() => choose(model)}
              />
            ))}
          </div>
        )}

        {data && data.ollama_reachable && (
          <div className="space-y-2 border-t border-border pt-3">
            <p className="text-xs font-medium text-muted-foreground">
              Install another model
            </p>
            {customPulls.map((m) => (
              <div key={m} className="space-y-1">
                <div className="font-mono text-xs text-foreground">{m}</div>
                <PullBar pull={pulls[m]} onCancel={() => stopPull(m)} />
              </div>
            ))}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                startPull(customModel);
                setCustomModel("");
              }}
              className="flex gap-2"
            >
              <Input
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="e.g. llama3.1:8b"
                className="h-9"
              />
              <Button
                type="submit"
                variant="outline"
                size="sm"
                disabled={!customModel.trim()}
              >
                <Download className="size-4" />
                Install
              </Button>
            </form>
            <p className="text-[11px] text-muted-foreground">
              Any Ollama model name. Downloads can be several GB.
            </p>
          </div>
        )}

        {error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
            {error}
          </div>
        )}

        {data && (
          <p className="text-xs text-muted-foreground">
            {data.source === "user"
              ? "Using your chosen model. "
              : "Auto-selected for this Mac. "}
            Changing the model takes effect on your next question.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function PullBar({ pull, onCancel }: { pull: PullState; onCancel: () => void }) {
  const known = pull.total > 0;
  const pct = known ? Math.min(100, (pull.completed / pull.total) * 100) : 30;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="truncate capitalize">{pull.status || "Downloading"}</span>
        {known && (
          <span className="shrink-0 tabular-nums">
            {fmtBytes(pull.completed)} / {fmtBytes(pull.total)}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
          <div
            className={cn(
              "h-full rounded-full bg-brand-gradient transition-all",
              !known && "animate-pulse",
            )}
            style={{ width: `${pct}%` }}
          />
        </div>
        <button
          type="button"
          onClick={onCancel}
          className="text-muted-foreground transition-colors hover:text-foreground"
          title="Cancel download"
        >
          <X className="size-3.5" />
        </button>
      </div>
    </div>
  );
}

interface RowProps {
  opt: ModelOption;
  recommended: boolean;
  saving: boolean;
  disabled: boolean;
  pull?: PullState;
  onSelect: () => void;
  onInstall: () => void;
  onCancelInstall: () => void;
}

function ModelRow({
  opt,
  recommended,
  saving,
  disabled,
  pull,
  onSelect,
  onInstall,
  onCancelInstall,
}: RowProps) {
  // Only installed models are selectable. Too-big / small-model choices are
  // allowed but flagged — user agency with a clear warning.
  const clickable = opt.installed && !disabled;
  const caveat = !opt.installed
    ? null
    : !opt.fits
      ? "Larger than recommended for your RAM — may run slowly or swap."
      : opt.warn;

  return (
    <div
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={clickable ? onSelect : undefined}
      onKeyDown={
        clickable
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect();
              }
            }
          : undefined
      }
      className={cn(
        "flex w-full flex-col gap-1.5 rounded-md border p-2.5 text-left transition-colors",
        opt.is_current ? "border-primary bg-primary/5" : "border-border",
        clickable && "cursor-pointer hover:bg-secondary/50",
      )}
    >
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "flex size-4 shrink-0 items-center justify-center rounded-full border",
            opt.is_current ? "border-primary bg-primary" : "border-muted-foreground/40",
          )}
        >
          {saving ? (
            <Loader2 className="size-3 animate-spin text-primary" />
          ) : opt.is_current ? (
            <Check className="size-3 text-primary-foreground" />
          ) : null}
        </span>
        <span className="font-mono text-sm">{opt.model}</span>
        <span className="text-xs text-muted-foreground">needs ~{gb(opt.needs_gb)}</span>
        <span className="ml-auto flex items-center gap-1">
          {recommended && <Badge tone="primary">Recommended</Badge>}
          {opt.is_current && <Badge tone="primary">Current</Badge>}
          {opt.installed ? (
            opt.fits ? (
              <Badge tone="muted">Installed</Badge>
            ) : (
              <Badge tone="amber">Too big</Badge>
            )
          ) : (
            <Badge tone="muted">Not installed</Badge>
          )}
        </span>
      </div>

      {pull ? (
        <div className="pl-6">
          <PullBar pull={pull} onCancel={onCancelInstall} />
        </div>
      ) : (
        !opt.installed && (
          <div className="pl-6">
            <Button
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onInstall();
              }}
              disabled={disabled}
            >
              <Download className="size-4" />
              Install
            </Button>
          </div>
        )
      )}

      {caveat && (
        <div className="flex items-start gap-1 pl-6 text-xs text-amber-600 dark:text-amber-400">
          <AlertTriangle className="mt-0.5 size-3 shrink-0" />
          <span>{caveat}</span>
        </div>
      )}
    </div>
  );
}

interface OtherRowProps {
  model: string;
  current: boolean;
  saving: boolean;
  disabled: boolean;
  onSelect: () => void;
}

function OtherModelRow({ model, current, saving, disabled, onSelect }: OtherRowProps) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onSelect}
      disabled={disabled}
      className={cn(
        "flex w-full items-center gap-2 rounded-md border p-2.5 text-left transition-colors",
        current ? "border-primary bg-primary/5" : "border-border hover:bg-secondary/50",
      )}
    >
      <span
        className={cn(
          "flex size-4 shrink-0 items-center justify-center rounded-full border",
          current ? "border-primary bg-primary" : "border-muted-foreground/40",
        )}
      >
        {saving ? (
          <Loader2 className="size-3 animate-spin text-primary" />
        ) : current ? (
          <Check className="size-3 text-primary-foreground" />
        ) : null}
      </span>
      <span className="font-mono text-sm">{model}</span>
      {current && (
        <span className="ml-auto">
          <Badge tone="primary">Current</Badge>
        </span>
      )}
    </button>
  );
}

function Badge({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone: "primary" | "muted" | "amber";
}) {
  return (
    <span
      className={cn(
        "rounded-full px-1.5 py-0.5 text-[10px] font-medium",
        tone === "primary" && "bg-primary/15 text-primary",
        tone === "muted" && "bg-muted text-muted-foreground",
        tone === "amber" && "bg-amber-500/15 text-amber-700 dark:text-amber-400",
      )}
    >
      {children}
    </span>
  );
}
