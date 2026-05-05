import { useEffect, useState } from "react";
import { Inbox, MessageCircle, Search, Settings as SettingsIcon } from "lucide-react";

import AccountSetup from "@/components/AccountSetup";
import SyncPanel from "@/components/SyncPanel";
import SearchView from "@/components/SearchView";
import ChatView from "@/components/ChatView";
import ModelStatusBadge from "@/components/ModelStatusBadge";

import { api, type AccountSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

type Tab = "search" | "chat" | "settings";

export default function App() {
  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [loadingAccounts, setLoadingAccounts] = useState(true);
  const [tab, setTab] = useState<Tab>("search");
  const [tickle, setTickle] = useState(0);

  const reload = async () => {
    setLoadingAccounts(true);
    try {
      const a = await api.listAccounts();
      setAccounts(a);
    } finally {
      setLoadingAccounts(false);
    }
  };

  useEffect(() => {
    reload();
  }, []);

  const noAccounts = !loadingAccounts && accounts.length === 0;

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Inbox className="size-5 text-primary" />
          <div className="text-sm font-semibold">MailFind</div>
        </div>
        <ModelStatusBadge />
      </header>

      <nav className="flex items-center gap-1 border-b border-border px-2 py-1">
        <TabButton
          active={tab === "search"}
          onClick={() => setTab("search")}
          icon={<Search className="size-4" />}
          label="Search"
        />
        <TabButton
          active={tab === "chat"}
          onClick={() => setTab("chat")}
          icon={<MessageCircle className="size-4" />}
          label="Ask"
        />
        <TabButton
          active={tab === "settings"}
          onClick={() => setTab("settings")}
          icon={<SettingsIcon className="size-4" />}
          label="Accounts"
        />
      </nav>

      <main className="flex-1 overflow-auto p-4">
        {noAccounts ? (
          <AccountSetup onAdded={reload} />
        ) : (
          <>
            {tab === "search" && <SearchView key={tickle} />}
            {tab === "chat" && <ChatView />}
            {tab === "settings" && (
              <div className="space-y-3">
                {accounts.map((a) => (
                  <SyncPanel
                    key={a.id}
                    account={a}
                    onChange={() => setTickle((n) => n + 1)}
                  />
                ))}
                <AccountSetup onAdded={reload} />
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

function TabButton({ active, onClick, icon, label }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors",
        active
          ? "bg-secondary text-secondary-foreground"
          : "text-muted-foreground hover:bg-secondary/60",
      )}
    >
      {icon}
      {label}
    </button>
  );
}
