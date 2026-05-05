import { useEffect, useState } from "react";
import { CheckCircle2, AlertTriangle } from "lucide-react";
import { api, type ModelStatus } from "@/lib/api";

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

  if (!status) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        Checking Ollama…
      </div>
    );
  }

  const ok =
    status.ollama_reachable &&
    status.embedding_available &&
    status.chat_available;
  const Icon = ok ? CheckCircle2 : AlertTriangle;
  return (
    <div
      className={`flex items-center gap-2 text-xs ${
        ok ? "text-green-400" : "text-yellow-400"
      }`}
    >
      <Icon className="size-4" />
      <span className="truncate">
        {status.ollama_reachable
          ? `${status.embedding_model}${
              status.embedding_available ? "" : " (missing)"
            } • ${status.chat_model}${
              status.chat_available ? "" : " (missing)"
            }`
          : `Ollama unreachable at ${status.endpoint}`}
      </span>
    </div>
  );
}
