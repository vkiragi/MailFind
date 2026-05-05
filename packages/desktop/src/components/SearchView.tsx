import { useState } from "react";
import { Search } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, type MessageHit } from "@/lib/api";

export default function SearchView() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MessageHit[]>([]);
  const [tookMs, setTookMs] = useState<number | null>(null);
  const [strategy, setStrategy] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const r = await api.search(query, 20);
      setResults(r.results);
      setTookMs(r.took_ms);
      setStrategy(
        [
          r.used_vector ? "vector" : null,
          r.used_keyword ? "keyword" : null,
        ]
          .filter(Boolean)
          .join(" + ") || "none",
      );
    } catch (err) {
      setError(String(err));
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="space-y-3">
      <form onSubmit={submit} className="flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your iCloud Mail…"
            className="pl-9"
          />
        </div>
        <Button type="submit" disabled={searching}>
          {searching ? "Searching…" : "Search"}
        </Button>
      </form>

      {tookMs !== null && (
        <div className="text-xs text-muted-foreground">
          {results.length} results in {tookMs}ms ({strategy})
        </div>
      )}

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
          {error}
        </div>
      )}

      <div className="space-y-2">
        {results.map((hit) => (
          <Card key={hit.message_id}>
            <CardContent className="space-y-1 p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="truncate text-sm font-semibold">
                  {hit.subject || "(no subject)"}
                </div>
                <div className="shrink-0 text-xs text-muted-foreground">
                  {new Date(hit.date).toLocaleString()}
                </div>
              </div>
              <div className="truncate text-xs text-muted-foreground">
                {hit.sender}
              </div>
              <div className="line-clamp-2 text-xs">{hit.snippet}</div>
              <div className="text-[10px] text-muted-foreground">
                score {hit.combined_score.toFixed(3)}
                {hit.similarity !== null
                  ? ` • sim ${hit.similarity.toFixed(3)}`
                  : ""}
                {hit.keyword_score !== null
                  ? ` • kw ${hit.keyword_score.toFixed(3)}`
                  : ""}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
