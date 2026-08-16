import { useEffect, useRef, useState } from "react";
import { Plus } from "lucide-react";

import AccountSetup from "@/components/AccountSetup";
import ModelPicker from "@/components/ModelPicker";
import SyncPanel from "@/components/SyncPanel";
import SearchView from "@/components/SearchView";
import ChatView from "@/components/ChatView";
import HelpView from "@/components/HelpView";
import Sidebar, { type Tab } from "@/components/Sidebar";
import { Button } from "@/components/ui/button";

import { api, type AccountSummary } from "@/lib/api";
import { computeAskGate } from "@/lib/askGate";

const VIEW_META: Record<Tab, { title: string; subtitle: string }> = {
  search: { title: "Search", subtitle: "Find mail by meaning, not just keywords" },
  chat: { title: "Ask", subtitle: "Ask questions and get cited answers from your mail" },
  settings: { title: "Accounts", subtitle: "Your accounts and the local Ask model" },
  help: { title: "Help", subtitle: "How MailFind works" },
};

export default function App() {
  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [loadingAccounts, setLoadingAccounts] = useState(true);
  const [tab, setTab] = useState<Tab>("search");
  const [tickle, setTickle] = useState(0);
  const [modelTick, setModelTick] = useState(0);
  const [showAddAccount, setShowAddAccount] = useState(false);
  // Set once the user manually picks a tab, so the capability-based default
  // below never overrides an explicit choice (e.g. a slow model check
  // resolving after they've already clicked into Search).
  const userPickedTab = useRef(false);
  const handleTab = (t: Tab) => {
    userPickedTab.current = true;
    setTab(t);
  };

  const reload = async () => {
    setLoadingAccounts(true);
    try {
      setAccounts(await api.listAccounts());
    } finally {
      setLoadingAccounts(false);
    }
  };

  useEffect(() => {
    reload();
  }, []);

  // Lead with Ask when this Mac can actually run it — that's the fuller
  // experience (Search's own retrieval, plus a synthesized, cited answer).
  // Search remains the default when Ask isn't viable (no capable model,
  // Ollama unreachable, etc.), since Search is the one thing guaranteed to
  // work on any machine. Only affects the initial landing tab.
  useEffect(() => {
    api
      .listModels()
      .then((models) => {
        if (userPickedTab.current) return;
        if (computeAskGate(models) === null) {
          setTab("chat");
        }
      })
      .catch(() => {
        // Can't determine capability (Ollama down, etc.) — stay on Search.
      });
  }, []);

  const noAccounts = !loadingAccounts && accounts.length === 0;
  // Help is documentation — keep it reachable even before an account exists.
  const showOnboarding = noAccounts && tab !== "help";
  const meta = VIEW_META[tab];

  return (
    <div className="flex h-full">
      <Sidebar tab={tab} onTab={handleTab} statusKey={modelTick} />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-8 py-4">
          <div>
            <h1 className="font-display text-lg font-semibold tracking-tight text-foreground">
              {showOnboarding ? "Welcome" : meta.title}
            </h1>
            <p className="text-sm text-muted-foreground">
              {showOnboarding
                ? "Let's connect your mail to get started"
                : meta.subtitle}
            </p>
          </div>
        </header>

        <main className="flex-1 overflow-auto">
          <div className="mx-auto max-w-3xl px-8 py-6">
            {showOnboarding ? (
              <AccountSetup onAdded={reload} />
            ) : (
              <>
                {tab === "search" && <SearchView key={tickle} />}
                {tab === "chat" && <ChatView />}
                {tab === "help" && <HelpView />}
                {tab === "settings" && (
                  <div className="space-y-4">
                    <ModelPicker onChange={() => setModelTick((n) => n + 1)} />
                    {accounts.map((a) => (
                      <SyncPanel
                        key={a.id}
                        account={a}
                        onChange={() => setTickle((n) => n + 1)}
                      />
                    ))}
                    {showAddAccount ? (
                      <AccountSetup
                        onAdded={() => {
                          setShowAddAccount(false);
                          reload();
                        }}
                      />
                    ) : (
                      <Button
                        variant="outline"
                        onClick={() => setShowAddAccount(true)}
                        className="w-full"
                      >
                        <Plus className="mr-1 size-4" />
                        Add another account
                      </Button>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
