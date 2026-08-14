import { Search, Sparkles, Inbox, Cpu, Lock, Command } from "lucide-react";

const STEPS = [
  {
    icon: Search,
    title: "Search by meaning",
    body: "Describe what you're looking for in plain words — 'receipts from last month', 'that flight booking'. Results rank by relevance, not just exact keywords. Click any result to open it in Apple Mail.",
  },
  {
    icon: Sparkles,
    title: "Ask your mail",
    body: "Ask a question and MailFind reads the most relevant emails, then answers with numbered citations. Click a citation to jump straight to the source email.",
  },
  {
    icon: Inbox,
    title: "Connect your accounts",
    body: "Add an iCloud account under Accounts using an app-specific password. Your first sync imports from Apple Mail instantly, then keeps up to date over IMAP.",
  },
  {
    icon: Cpu,
    title: "Pick the right model",
    body: "The Ask model is chosen automatically to fit your Mac's memory. Change it anytime under Accounts → Ask model. Search works even without a model.",
  },
  {
    icon: Lock,
    title: "Private by design",
    body: "Your mail, the search index, and the language model all stay on your Mac. The only thing that leaves the device is the secure fetch to your mail provider.",
  },
  {
    icon: Command,
    title: "Handy shortcuts",
    body: "Zoom the whole app with ⌘+, ⌘-, and ⌘0 (also under the View menu). MailFind follows your macOS light or dark appearance automatically.",
  },
];

export default function HelpView() {
  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-border bg-card p-6 shadow-soft">
        <div className="flex items-center gap-3">
          <div className="flex size-11 items-center justify-center rounded-xl bg-brand-gradient text-white">
            <Sparkles className="size-6" />
          </div>
          <div>
            <h2 className="font-display text-xl font-semibold text-foreground">
              Getting started
            </h2>
            <p className="text-sm text-muted-foreground">
              Private search and Q&amp;A over your Apple Mail — here's how it works.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-2.5">
        {STEPS.map(({ icon: Icon, title, body }) => (
          <div
            key={title}
            className="flex gap-4 rounded-xl border border-border bg-card p-4 shadow-soft"
          >
            <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Icon className="size-5" />
            </div>
            <div className="min-w-0">
              <h3 className="font-medium text-foreground">{title}</h3>
              <p className="mt-0.5 text-sm leading-relaxed text-muted-foreground">
                {body}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
