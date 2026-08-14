import { useEffect, useMemo, useRef, useState } from "react";
import { Send, ChevronDown, AlertTriangle, RefreshCw, Sparkles } from "lucide-react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { relativeTime } from "@/lib/format";
import {
  api,
  openInMail,
  ASK_TOKEN_EVENT,
  type AnswerCitation,
  type AnswerResponse,
  type ModelList,
} from "@/lib/api";

const EXAMPLES = [
  "What subscriptions am I paying for?",
  "Summarize my recent online orders",
  "Do I have any upcoming travel?",
];

// For the streaming preview, before citations are known.
const NO_CITATIONS = new Set<number>();
const noop = () => {};

/** Describes why Ask is unavailable, plus the fix. `null` means Ask is ready. */
interface AskGate {
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
 */
function computeGate(m: ModelList | null): AskGate | null {
  if (!m) return null; // still loading — don't gate yet
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

export default function ChatView() {
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [streamText, setStreamText] = useState("");
  const [history, setHistory] = useState<AnswerResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelList | null>(null);

  const checkModels = async () => {
    try {
      setModels(await api.listModels());
    } catch {
      setModels(null);
    }
  };

  // Re-runs on mount, which includes every time the user switches to the Ask
  // tab (App unmounts ChatView on other tabs), so a model change in the picker
  // is reflected without extra wiring.
  useEffect(() => {
    checkModels();
  }, []);

  const gate = computeGate(models);

  const runAsk = async (q: string) => {
    const text = q.trim();
    if (!text || pending) return;
    setQuestion("");
    setPendingQuestion(text);
    setStreamText("");
    setPending(true);
    setError(null);
    let unlisten: UnlistenFn | undefined;
    try {
      // Subscribe before invoking so no early fragments are missed. Retrieval
      // runs before the first token, so this is registered well in time.
      unlisten = await listen<string>(ASK_TOKEN_EVENT, (ev) => {
        setStreamText((prev) => prev + ev.payload);
      });
      const r = await api.ask(text, 8);
      setHistory((prev) => [r, ...prev]);
    } catch (err) {
      setError(String(err));
    } finally {
      unlisten?.();
      setPending(false);
      setStreamText("");
    }
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    runAsk(question);
  };

  const showEmpty = !gate && !pending && !error && history.length === 0;

  return (
    <div className="space-y-4">
      {gate ? (
        <GateCard gate={gate} onRecheck={checkModels} />
      ) : (
        <form onSubmit={submit} className="flex gap-2">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your mail…"
            autoFocus
            className="h-12 text-base"
          />
          <Button type="submit" variant="brand" size="lg" disabled={pending}>
            <Send className="size-4" />
            {pending ? "Thinking…" : "Ask"}
          </Button>
        </form>
      )}

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {showEmpty && <AskEmpty onPick={runAsk} />}

      <div className="space-y-3">
        {pending && (
          <Card>
            <CardContent className="space-y-3 p-4">
              <UserBubble text={pendingQuestion} />
              {streamText ? (
                <div className="space-y-2.5 text-[15px] leading-relaxed text-foreground">
                  <MarkdownAnswer
                    text={streamText}
                    referenced={NO_CITATIONS}
                    jumpTo={noop}
                  />
                  <span className="inline-block h-4 w-[3px] animate-pulse rounded-full bg-primary align-text-bottom" />
                </div>
              ) : (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span className="size-2 animate-pulse rounded-full bg-primary" />
                  Searching your mail…
                </div>
              )}
            </CardContent>
          </Card>
        )}
        {history.map((entry, idx) => (
          <AnswerExchange key={idx} entry={entry} />
        ))}
      </div>
    </div>
  );
}

function AskEmpty({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-8 text-center shadow-soft">
      <div className="mx-auto flex size-12 items-center justify-center rounded-xl bg-brand-gradient text-white shadow-brand">
        <Sparkles className="size-6" />
      </div>
      <h2 className="mt-4 font-display text-lg font-semibold text-foreground">
        Ask anything about your mail
      </h2>
      <p className="mx-auto mt-1 max-w-sm text-sm text-muted-foreground">
        MailFind finds the relevant emails and answers with citations you can open.
      </p>
      <div className="mx-auto mt-5 flex max-w-md flex-col gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => onPick(ex)}
            className="rounded-lg border border-border bg-secondary/50 px-3.5 py-2.5 text-left text-sm text-foreground transition-colors hover:border-primary/40 hover:bg-primary/10 hover:text-primary"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}

function GateCard({ gate, onRecheck }: { gate: AskGate; onRecheck: () => void }) {
  return (
    <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4">
      <div className="flex items-start gap-2.5">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-500" />
        <div className="space-y-1.5">
          <div className="text-sm font-semibold text-foreground">{gate.title}</div>
          <p className="text-sm text-muted-foreground">{gate.body}</p>
          {gate.pull && (
            <p className="text-sm text-muted-foreground">
              Install it with:{" "}
              <code className="rounded bg-background/60 px-1.5 py-0.5 font-mono text-xs">
                ollama pull {gate.pull}
              </code>
            </p>
          )}
          <Button variant="outline" size="sm" onClick={onRecheck} className="mt-1">
            <RefreshCw className="mr-1 size-4" />
            Recheck
          </Button>
        </div>
      </div>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-primary/10 px-3.5 py-2 text-sm text-foreground">
        {text}
      </div>
    </div>
  );
}

function AnswerExchange({ entry }: { entry: AnswerResponse }) {
  const [showOthers, setShowOthers] = useState(false);
  const [flash, setFlash] = useState<number | null>(null);
  const cardRefs = useRef<Record<number, HTMLDivElement | null>>({});

  const citations = entry.citations;

  // Which [n] markers actually appear in the answer text (and map to a real
  // citation). These are the emails the model leaned on; the rest were just
  // retrieved candidates.
  const referenced = useMemo(() => {
    const set = new Set<number>();
    for (const m of entry.answer.matchAll(/\[(\d+)\]/g)) {
      const n = Number(m[1]);
      if (n >= 1 && n <= citations.length) set.add(n);
    }
    return set;
  }, [entry.answer, citations.length]);

  const jumpTo = (n: number) => {
    const el = cardRefs.current[n];
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlash(n);
    window.setTimeout(() => setFlash((cur) => (cur === n ? null : cur)), 1500);
  };

  const all = citations.map((c, i) => ({ c, n: i + 1 }));
  const cited = all.filter((x) => referenced.has(x.n));
  const hasCited = cited.length > 0;
  const primary = hasCited ? cited : all;
  const extra = hasCited ? all.filter((x) => !referenced.has(x.n)) : [];

  return (
    <Card className="animate-fade-up">
      <CardContent className="space-y-4 p-4">
        <UserBubble text={entry.question} />

        <div className="space-y-2.5 text-[15px] leading-relaxed text-foreground">
          <MarkdownAnswer text={entry.answer} referenced={referenced} jumpTo={jumpTo} />
        </div>

        {citations.length > 0 && (
          <div className="space-y-2 border-t border-border pt-3">
            <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              {hasCited ? "Sources" : "Emails searched"}
            </div>
            <div className="space-y-1.5">
              {primary.map(({ c, n }) => (
                <SourceCard
                  key={c.message_id}
                  citation={c}
                  n={n}
                  flash={flash === n}
                  setRef={(el) => (cardRefs.current[n] = el)}
                />
              ))}
            </div>

            {extra.length > 0 && (
              <>
                <button
                  type="button"
                  onClick={() => setShowOthers((v) => !v)}
                  className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
                >
                  <ChevronDown
                    className={cn(
                      "size-3 transition-transform",
                      showOthers && "rotate-180",
                    )}
                  />
                  {showOthers ? "Hide" : "Other emails searched"} ({extra.length})
                </button>
                {showOthers && (
                  <div className="space-y-1.5">
                    {extra.map(({ c, n }) => (
                      <SourceCard
                        key={c.message_id}
                        citation={c}
                        n={n}
                        flash={flash === n}
                        setRef={(el) => (cardRefs.current[n] = el)}
                      />
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        <div className="text-[11px] text-muted-foreground">
          <span className="font-mono">{entry.model}</span> ·{" "}
          {(entry.took_ms / 1000).toFixed(1)}s
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Renders the model's answer as light Markdown — bold, bullet/numbered lists,
 * and paragraphs — so the raw `**` / `-` syntax never shows. `[n]` markers that
 * map to a real citation become clickable chips that scroll to that source.
 * (Deliberately minimal: no dependency, and full control over the chips.)
 */
function MarkdownAnswer({
  text,
  referenced,
  jumpTo,
}: {
  text: string;
  referenced: Set<number>;
  jumpTo: (n: number) => void;
}) {
  const blocks = parseBlocks(text);
  return (
    <>
      {blocks.map((b, i) => {
        if (b.type === "heading") {
          return (
            <p key={i} className="font-semibold text-foreground">
              {renderInline(b.text, referenced, jumpTo)}
            </p>
          );
        }
        if (b.type === "list") {
          const Tag = b.ordered ? "ol" : "ul";
          return (
            <Tag
              key={i}
              className={cn(
                "space-y-1 pl-5",
                b.ordered ? "list-decimal" : "list-disc",
              )}
            >
              {b.items.map((it, j) => (
                <li key={j} className="pl-1 marker:text-muted-foreground">
                  {renderInline(it, referenced, jumpTo)}
                </li>
              ))}
            </Tag>
          );
        }
        return <p key={i}>{renderInline(b.text, referenced, jumpTo)}</p>;
      })}
    </>
  );
}

type Block =
  | { type: "paragraph"; text: string }
  | { type: "heading"; text: string }
  | { type: "list"; ordered: boolean; items: string[] };

function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flushPara = () => {
    if (para.length) {
      blocks.push({ type: "paragraph", text: para.join(" ") });
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      blocks.push({ type: "list", ordered: list.ordered, items: list.items });
      list = null;
    }
  };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) {
      flushPara();
      flushList();
      continue;
    }
    const bullet = line.match(/^[-*•]\s+(.*)/);
    const numbered = line.match(/^\d+[.)]\s+(.*)/);
    const heading = line.match(/^#{1,6}\s+(.*)/);
    if (bullet) {
      flushPara();
      if (!list || list.ordered) {
        flushList();
        list = { ordered: false, items: [] };
      }
      list.items.push(bullet[1]);
    } else if (numbered) {
      flushPara();
      if (!list || !list.ordered) {
        flushList();
        list = { ordered: true, items: [] };
      }
      list.items.push(numbered[1]);
    } else if (heading) {
      flushPara();
      flushList();
      blocks.push({ type: "heading", text: heading[1] });
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return blocks;
}

/** Inline formatting: **bold** and clickable [n] citation chips. */
function renderInline(
  text: string,
  referenced: Set<number>,
  jumpTo: (n: number) => void,
): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const re = /\*\*(.+?)\*\*|\[(\d+)\]/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[1] !== undefined) {
      nodes.push(
        <strong key={`b${key++}`} className="font-semibold text-foreground">
          {m[1]}
        </strong>,
      );
    } else {
      const n = Number(m[2]);
      if (referenced.has(n)) {
        nodes.push(
          <button
            key={`c${key++}`}
            type="button"
            onClick={() => jumpTo(n)}
            className="mx-0.5 inline-flex h-[1.15rem] min-w-[1.15rem] items-center justify-center rounded-md bg-primary/15 px-1 align-text-top text-[10px] font-semibold text-primary transition-colors hover:bg-primary/30"
            title={`Jump to source ${n}`}
          >
            {n}
          </button>,
        );
      } else {
        nodes.push(m[0]);
      }
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function SourceCard({
  citation: c,
  n,
  flash,
  setRef,
}: {
  citation: AnswerCitation;
  n: number;
  flash: boolean;
  setRef: (el: HTMLDivElement | null) => void;
}) {
  const clickable = !!c.rfc822_message_id;
  return (
    <div
      ref={setRef}
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={() => openInMail(c.rfc822_message_id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openInMail(c.rfc822_message_id);
        }
      }}
      className={cn(
        "flex gap-3 rounded-lg border border-border p-2.5 text-xs transition-colors",
        clickable && "cursor-pointer hover:bg-secondary/60",
        flash && "ring-2 ring-primary",
      )}
      title={clickable ? "Open in Apple Mail" : undefined}
    >
      <div className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-semibold text-primary">
        {n}
      </div>
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="truncate font-semibold text-foreground">{c.subject}</div>
        <div className="truncate text-muted-foreground">
          {c.sender ? `${c.sender} · ` : ""}
          {relativeTime(c.date)}
        </div>
        <div className="line-clamp-1 text-muted-foreground">{c.snippet}</div>
      </div>
    </div>
  );
}
