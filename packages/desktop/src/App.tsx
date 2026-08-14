import { useEffect, useState } from "react";
import { Plus } from "lucide-react";

import AccountSetup from "@/components/AccountSetup";
import ModelPicker from "@/components/ModelPicker";
import SyncPanel from "@/components/SyncPanel";
import SearchView from "@/components/SearchView";
import ChatView from "@/components/ChatView";
import Sidebar, { type Tab } from "@/components/Sidebar";
import { Button } from "@/components/ui/button";

import { api, type AccountSummary } from "@/lib/api";

const VIEW_META: Record<Tab, { title: string; subtitle: string }> = {
  search: { title: "Search", subtitle: "Find mail by meaning, not just keywords" },
  chat: { title: "Ask", subtitle: "Ask questions and get cited answers from your mail" },
  settings: { title: "Accounts", subtitle: "Your accounts and the local Ask model" },
};

export default function App() {
  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [loadingAccounts, setLoadingAccounts] = useState(true);
  const [tab, setTab] = useState<Tab>("search");
  const [tickle, setTickle] = useState(0);
  const [modelTick, setModelTick] = useState(0);
  const [showAddAccount, setShowAddAccount] = useState(false);

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

  const noAccounts = !loadingAccounts && accounts.length === 0;
  const meta = VIEW_META[tab];

  return (
    <div className="flex h-full">
      <Sidebar tab={tab} onTab={setTab} statusKey={modelTick} />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-8 py-4">
          <div>
            <h1 className="font-display text-lg font-semibold tracking-tight text-foreground">
              {noAccounts ? "Welcome" : meta.title}
            </h1>
            <p className="text-sm text-muted-foreground">
              {noAccounts ? "Let's connect your mail to get started" : meta.subtitle}
            </p>
          </div>
        </header>

        <main className="flex-1 overflow-auto">
          <div className="mx-auto max-w-3xl px-8 py-6">
            {noAccounts ? (
              <AccountSetup onAdded={reload} />
            ) : (
              <>
                {tab === "search" && <SearchView key={tickle} />}
                {tab === "chat" && <ChatView />}
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
