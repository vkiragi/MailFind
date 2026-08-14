import { useEffect, useState } from "react";
import { AlertTriangle, Check, Cpu, Loader2, RefreshCw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api, type ModelList, type ModelOption } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  /** Notify parent so it can refresh the header model badge after a change. */
  onChange?: () => void;
}

function gb(n: number): string {
  return `${Math.round(n)} GB`;
}

export default function ModelPicker({ onChange }: Props) {
  const [data, setData] = useState<ModelList | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
              Detected {gb(data.total_ram_gb)} RAM (~{gb(data.budget_gb)} free
              for a model).
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
            model, so Ask is off — search still works. You can still opt into a
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
                onSelect={() => choose(opt.model)}
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

interface RowProps {
  opt: ModelOption;
  recommended: boolean;
  saving: boolean;
  disabled: boolean;
  onSelect: () => void;
}

function ModelRow({ opt, recommended, saving, disabled, onSelect }: RowProps) {
  // A model can be chosen only once it's installed. Too-big / small-model
  // choices are allowed but flagged — user agency with a clear warning.
  const selectable = opt.installed;
  const caveat = !opt.installed
    ? null
    : !opt.fits
      ? "Larger than recommended for your RAM — may run slowly or swap."
      : opt.warn;

  return (
    <button
      type="button"
      onClick={selectable && !disabled ? onSelect : undefined}
      disabled={!selectable || disabled}
      className={cn(
        "flex w-full flex-col gap-1 rounded-md border p-2.5 text-left transition-colors",
        opt.is_current
          ? "border-primary bg-primary/5"
          : "border-border hover:bg-secondary/50",
        !selectable && "cursor-default opacity-70 hover:bg-transparent",
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
      {!opt.installed && (
        <div className="pl-6 text-xs text-muted-foreground">
          Install to use:{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono">
            ollama pull {opt.model}
          </code>
        </div>
      )}
      {caveat && (
        <div className="flex items-start gap-1 pl-6 text-xs text-amber-600 dark:text-amber-400">
          <AlertTriangle className="mt-0.5 size-3 shrink-0" />
          <span>{caveat}</span>
        </div>
      )}
    </button>
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
        current
          ? "border-primary bg-primary/5"
          : "border-border hover:bg-secondary/50",
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
        tone === "amber" &&
          "bg-amber-500/15 text-amber-700 dark:text-amber-400",
      )}
    >
      {children}
    </span>
  );
}
