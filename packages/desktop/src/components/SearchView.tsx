import { useState } from "react";
import { Search, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar } from "@/components/ui/avatar";
import { api, openInMail, type MessageHit } from "@/lib/api";
import { relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";

const EXAMPLES = [
  "receipts and invoices",
  "flight confirmations",
  "password reset emails",
  "subscription renewals",
];

export default function SearchView() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MessageHit[]>([]);
  const [tookMs, setTookMs] = useState<number | null>(null);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (q: string) => {
    if (!q.trim()) return;
    setSearching(true);
    setError(null);
    setSearched(true);
    try {
      const r = await api.search(q, 20);
      setResults(r.results);
      setTookMs(r.took_ms);
    } catch (err) {
      setError(String(err));
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    run(query);
  };

  const pickExample = (q: string) => {
    setQuery(q);
    run(q);
  };

  return (
    <div className="space-y-5">
      <form onSubmit={submit} className="flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 size-[18px] -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your mail by meaning…"
            autoFocus
            className="h-12 pl-11 text-base"
          />
        </div>
        <Button type="submit" variant="brand" size="lg" disabled={searching}>
          {searching ? "Searching…" : "Search"}
        </Button>
      </form>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {!searched && !error && (
        <div className="rounded-2xl border border-border bg-card p-8 text-center shadow-soft">
          <div className="mx-auto flex size-12 items-center justify-center rounded-xl bg-brand-gradient text-white">
            <Sparkles className="size-6" />
          </div>
          <h2 className="mt-4 font-display text-lg font-semibold text-foreground">
            Search your mail by meaning
          </h2>
          <p className="mx-auto mt-1 max-w-sm text-sm text-muted-foreground">
            Describe what you're looking for in plain language — MailFind ranks by
            relevance, not just exact words.
          </p>
          <div className="mt-5 flex flex-wrap justify-center gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => pickExample(ex)}
                className="rounded-full border border-border bg-secondary/60 px-3 py-1.5 text-sm text-foreground transition-colors hover:border-primary/40 hover:bg-primary/10 hover:text-primary"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      {searched && !searching && !error && (
        <div className="text-xs text-muted-foreground">
          {results.length === 0
            ? "No matches found"
            : `${results.length} result${results.length === 1 ? "" : "s"}${
                tookMs !== null ? ` · ${tookMs}ms` : ""
              }`}
        </div>
      )}

      {searching && (
        <div className="space-y-2.5">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="flex gap-3 rounded-xl border border-border bg-card p-3.5 shadow-soft"
            >
              <div className="size-9 shrink-0 animate-pulse rounded-full bg-muted" />
              <div className="flex-1 space-y-2 py-0.5">
                <div className="h-3.5 w-2/3 animate-pulse rounded bg-muted" />
                <div className="h-3 w-1/3 animate-pulse rounded bg-muted" />
                <div className="h-3 w-full animate-pulse rounded bg-muted" />
              </div>
            </div>
          ))}
        </div>
      )}

      {!searching && results.length > 0 && (
        <div className="space-y-2.5">
          {results.map((hit, i) => (
            <ResultRow key={hit.message_id} hit={hit} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}

function ResultRow({ hit, index }: { hit: MessageHit; index: number }) {
  const clickable = !!hit.rfc822_message_id;
  return (
    <button
      type="button"
      onClick={() => openInMail(hit.rfc822_message_id)}
      disabled={!clickable}
      style={{ animationDelay: `${Math.min(index, 10) * 30}ms` }}
      className={cn(
        "flex w-full animate-fade-up gap-3 rounded-xl border border-border bg-card p-3.5 text-left shadow-soft transition-all",
        clickable
          ? "hover:-translate-y-px hover:shadow-card"
          : "cursor-default",
      )}
      title={clickable ? "Open in Apple Mail" : undefined}
    >
      <Avatar name={hit.sender || hit.sender_email || "?"} />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-3">
          <div className="truncate font-medium text-foreground">
            {hit.subject || "(no subject)"}
          </div>
          <div className="shrink-0 text-xs text-muted-foreground">
            {relativeTime(hit.date)}
          </div>
        </div>
        <div className="truncate text-sm text-muted-foreground">{hit.sender}</div>
        {hit.snippet && (
          <div className="mt-1 line-clamp-2 text-sm text-muted-foreground/90">
            {hit.snippet}
          </div>
        )}
        <div className="mt-2">
          <MatchPill hit={hit} />
        </div>
      </div>
    </button>
  );
}

function MatchPill({ hit }: { hit: MessageHit }) {
  const semantic = hit.similarity !== null;
  const keyword = hit.keyword_score !== null;
  const label = semantic && keyword ? "semantic + keyword" : semantic ? "semantic match" : "keyword match";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium",
        semantic
          ? "bg-primary/10 text-primary"
          : "bg-secondary text-muted-foreground",
      )}
    >
      {semantic && <Sparkles className="size-3" />}
      {label}
    </span>
  );
}
