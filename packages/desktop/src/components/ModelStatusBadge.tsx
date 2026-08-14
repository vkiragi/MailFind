import { useEffect, useState } from "react";
import { api, type ModelStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function ModelStatusBadge() {
  const [status, setStatus] = useState<ModelStatus | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const s = await api.modelStatus();
        if (active) setStatus(s);
      } catch {
        if (active) setStatus(null);
      }
    };
    refresh();
    const t = setInterval(refresh, 8000);
    return () => {
      active = false;
      clearInterval(t);
    };
  }, []);

  const { dot, primary, secondary } = describe(status);

  return (
    <div className="flex items-center gap-2.5 rounded-lg border border-border bg-secondary/50 px-3 py-2.5">
      <span className={cn("size-2 shrink-0 rounded-full", dot)} />
      <div className="min-w-0 leading-tight">
        <div className="truncate font-mono text-xs text-foreground">{primary}</div>
        <div className="truncate text-[11px] text-muted-foreground">{secondary}</div>
      </div>
    </div>
  );
}

function describe(status: ModelStatus | null): {
  dot: string;
  primary: string;
  secondary: string;
} {
  if (!status) {
    return { dot: "bg-muted-foreground animate-pulse", primary: "Ollama", secondary: "Connecting…" };
  }
  if (!status.ollama_reachable) {
    return { dot: "bg-amber-500", primary: "Ollama offline", secondary: "Search still works" };
  }
  const ok = status.embedding_available && status.chat_available;
  return {
    dot: ok ? "bg-emerald-500" : "bg-amber-500",
    primary: status.chat_model || "No model",
    secondary: ok
      ? "Running locally"
      : status.chat_available
        ? "Embedder missing"
        : "Needs a model",
  };
}
