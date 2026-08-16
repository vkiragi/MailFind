import type { ModelList } from "@/lib/api";

/** Describes why Ask is unavailable, plus the fix. `null` means Ask is ready. */
export interface AskGate {
  title: string;
  body: string;
  /** A model to `ollama pull`, when the fix is installing one. */
  pull?: string;
}

/**
 * Decides whether the Ask tab can serve questions. Search always works; Ask
 * needs a reachable Ollama with the active chat model actually installed. On a
 * machine below the Ask RAM floor (auto-pick says search-only) we keep Ask off
 * unless the user explicitly opted into a small model via the picker.
 *
 * `m === null` means the model list hasn't loaded yet — returns `null` (no
 * gate shown) so ChatView doesn't flash an error before data arrives. Callers
 * that need to distinguish "ready" from "still loading" (e.g. picking a
 * default landing tab) should check `m !== null` themselves before trusting
 * a `null` result as "ready".
 */
export function computeAskGate(m: ModelList | null): AskGate | null {
  if (!m) return null;
  if (!m.ollama_reachable) {
    return {
      title: "Ask is off — Ollama isn't running",
      body: "The Ask tab needs a local model served by Ollama. Your searches still work without it.",
    };
  }
  if (m.auto_pick_state === "search_only" && m.source !== "user") {
    return {
      title: `Ask is off — ${Math.round(m.total_ram_gb)} GB is below the threshold for a reliable chat model`,
      body: "Semantic search runs on any Mac. To try Ask anyway, pick the small model under Accounts → Ask model.",
    };
  }
  const currentInstalled =
    m.options.some((o) => o.is_current && o.installed) ||
    m.other_installed.includes(m.current_model);
  if (!currentInstalled) {
    return {
      title: "Ask needs a model installed",
      body: "Install a local chat model, then recheck. You can also choose a different one under Accounts → Ask model.",
      pull: m.auto_pick_model ?? m.current_model,
    };
  }
  return null;
}
