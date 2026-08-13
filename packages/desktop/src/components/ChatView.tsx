import { useMemo, useRef, useState } from "react";
import { Send, ChevronDown } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import {
  api,
  openInMail,
  type AnswerCitation,
  type AnswerResponse,
} from "@/lib/api";

export default function ChatView() {
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [history, setHistory] = useState<AnswerResponse[]>([]);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    const q = question;
    setQuestion("");
    setPendingQuestion(q);
    setPending(true);
    setError(null);
    try {
      const r = await api.ask(q, 8);
      setHistory((prev) => [r, ...prev]);
    } catch (err) {
      setError(String(err));
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="space-y-3">
      <form onSubmit={submit} className="flex gap-2">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your mail…"
        />
        <Button type="submit" disabled={pending}>
          <Send className="mr-1" />
          {pending ? "Thinking…" : "Ask"}
        </Button>
      </form>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
          {error}
        </div>
      )}

      <div className="space-y-3">
        {pending && (
          <Card>
            <CardContent className="space-y-3 p-4">
              <UserBubble text={pendingQuestion} />
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="size-2 animate-pulse rounded-full bg-primary" />
                Searching your mail…
              </div>
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

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-muted px-3.5 py-2 text-sm">
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
    <Card>
      <CardContent className="space-y-4 p-4">
        <UserBubble text={entry.question} />

        <div className="whitespace-pre-wrap text-sm leading-relaxed">
          {renderAnswer(entry.answer, referenced, jumpTo)}
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

        <div className="text-[10px] text-muted-foreground">
          {entry.model} • {(entry.took_ms / 1000).toFixed(1)}s
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Renders the answer text, replacing each `[n]` citation marker with a small
 * clickable chip that scrolls to source `n`. Markers that don't map to a real
 * citation are left as plain text.
 */
function renderAnswer(
  answer: string,
  referenced: Set<number>,
  jumpTo: (n: number) => void,
) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  let key = 0;
  for (const m of answer.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    const start = m.index ?? 0;
    if (start > last) parts.push(answer.slice(last, start));
    if (referenced.has(n)) {
      parts.push(
        <button
          key={`chip-${key++}`}
          type="button"
          onClick={() => jumpTo(n)}
          className="mx-0.5 inline-flex h-[1.1rem] min-w-[1.1rem] items-center justify-center rounded bg-primary/15 px-1 align-text-top text-[10px] font-semibold text-primary transition-colors hover:bg-primary/30"
          title={`Jump to source ${n}`}
        >
          {n}
        </button>,
      );
    } else {
      parts.push(m[0]);
    }
    last = start + m[0].length;
  }
  if (last < answer.length) parts.push(answer.slice(last));
  return parts;
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
        "flex gap-3 rounded-md border border-border p-2.5 text-xs transition-colors",
        clickable && "cursor-pointer hover:bg-muted/40",
        flash && "ring-2 ring-primary",
      )}
      title={clickable ? "Open in Apple Mail" : undefined}
    >
      <div className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[11px] font-semibold text-muted-foreground">
        {n}
      </div>
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="truncate font-semibold text-foreground">{c.subject}</div>
        <div className="truncate text-muted-foreground">
          {c.sender ? `${c.sender} • ` : ""}
          {new Date(c.date).toLocaleDateString()}
        </div>
        <div className="line-clamp-1 text-muted-foreground">{c.snippet}</div>
      </div>
    </div>
  );
}
