import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";

interface Props {
  onAdded: () => void;
}

export default function AccountSetup({ onAdded }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [host, setHost] = useState("imap.mail.me.com");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await api.addAccount({
        email,
        password,
        imap_host: host,
        imap_port: 993,
      });
      setEmail("");
      setPassword("");
      onAdded();
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Connect your iCloud mail</CardTitle>
        <p className="text-xs text-muted-foreground">
          Generate an{" "}
          <a
            href="https://support.apple.com/en-us/102654"
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            app-specific password
          </a>{" "}
          at appleid.apple.com. Credentials are stored in your macOS keychain.
        </p>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="space-y-3">
          <div>
            <label className="mb-1 block text-xs text-muted-foreground">
              iCloud email
            </label>
            <Input
              type="email"
              required
              autoComplete="username"
              placeholder="you@icloud.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-muted-foreground">
              App-specific password
            </label>
            <Input
              type="password"
              required
              autoComplete="current-password"
              placeholder="xxxx-xxxx-xxxx-xxxx"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-muted-foreground">
              IMAP host
            </label>
            <Input
              type="text"
              value={host}
              onChange={(e) => setHost(e.target.value)}
            />
          </div>
          {error && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
              {error}
            </div>
          )}
          <Button type="submit" variant="brand" disabled={submitting}>
            {submitting ? "Connecting…" : "Add account"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
