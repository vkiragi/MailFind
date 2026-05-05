import { useState } from "react";
import { Send } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, type AnswerResponse } from "@/lib/api";

export default function ChatView() {
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState(false);
  const [history, setHistory] = useState<AnswerResponse[]>([]);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    const q = question;
    setQuestion("");
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
        {history.map((entry, idx) => (
          <Card key={idx}>
            <CardContent className="space-y-3 p-4">
              <div className="text-xs font-semibold text-muted-foreground">
                Q
              </div>
              <div className="text-sm">{entry.question}</div>
              <div className="text-xs font-semibold text-muted-foreground">
                {entry.model}
              </div>
              <div className="whitespace-pre-wrap text-sm">{entry.answer}</div>
              {entry.citations.length > 0 && (
                <div className="space-y-1 border-t border-border pt-2">
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                    Citations
                  </div>
                  {entry.citations.map((c) => (
                    <div
                      key={c.message_id}
                      className="rounded-md border border-border p-2 text-xs"
                    >
                      <div className="truncate font-semibold">{c.subject}</div>
                      <div className="truncate text-muted-foreground">
                        {c.sender} • {new Date(c.date).toLocaleString()}
                      </div>
                      <div className="line-clamp-2 text-muted-foreground">
                        {c.snippet}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div className="text-[10px] text-muted-foreground">
                {entry.took_ms}ms
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
