import { invoke } from "@tauri-apps/api/core";
import { open as openExternal } from "@tauri-apps/plugin-shell";

/**
 * Opens a message in Apple Mail via the `message:` URL scheme. macOS registers
 * Mail.app as the handler by default; if the user has a different mail client,
 * it'll route there instead. Falls back silently if the Message-ID is missing
 * or no handler is registered.
 */
export async function openInMail(rfc822MessageId: string | null): Promise<void> {
  if (!rfc822MessageId) return;
  // Strip any surrounding angle brackets — Message-IDs are sometimes stored as
  // `<foo@example.com>` but the URL scheme wants the raw value.
  const id = rfc822MessageId.replace(/^<|>$/g, "");
  try {
    await openExternal(`message:%3C${encodeURIComponent(id)}%3E`);
  } catch {
    // Fallback for clients that don't expect the URL-encoded brackets.
    try {
      await openExternal(`message:${id}`);
    } catch {
      // No handler / scheme rejected — nothing we can do client-side.
    }
  }
}

export interface AccountSummary {
  id: string;
  email: string;
  display_name: string | null;
  imap_host: string;
  created_at: string;
}

export interface SyncStatus {
  account_id: string | null;
  is_running: boolean;
  last_sync: string | null;
  total_messages: number;
  embedded_messages: number;
  last_error: string | null;
}

export interface ModelStatus {
  ollama_reachable: boolean;
  embedding_model: string;
  embedding_available: boolean;
  chat_model: string;
  chat_available: boolean;
  endpoint: string;
}

export interface MessageHit {
  message_id: string;
  account_id: string;
  thread_id: string | null;
  rfc822_message_id: string | null;
  subject: string;
  sender: string;
  sender_email: string | null;
  recipients: string;
  date: string;
  snippet: string;
  body_preview: string;
  similarity: number | null;
  keyword_score: number | null;
  combined_score: number;
}

export interface SearchResponse {
  query: string;
  results: MessageHit[];
  total: number;
  used_vector: boolean;
  used_keyword: boolean;
  took_ms: number;
}

export interface AnswerCitation {
  message_id: string;
  rfc822_message_id: string | null;
  subject: string;
  sender: string;
  date: string;
  snippet: string;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  model: string;
  citations: AnswerCitation[];
  took_ms: number;
}

export interface AddAccountRequest {
  email: string;
  password: string;
  display_name?: string | null;
  imap_host?: string | null;
  imap_port?: number | null;
}

export type SyncWindow =
  | "day"
  | "week"
  | "two_weeks"
  | "month"
  | "three_months";

export type SyncPhase =
  | "connecting"
  | "searching"
  | "fetching"
  | "storing"
  | "done"
  | "failed";

export interface SyncProgress {
  account_id: string;
  phase: SyncPhase;
  total: number | null;
  current: number;
  message: string | null;
}

export const SYNC_PROGRESS_EVENT = "sync:progress";

export interface IngestFixtureRequest {
  account_id: string;
  path: string;
}

export interface IngestFixtureResponse {
  imported: number;
  skipped: number;
  errors: string[];
}

export interface AppleMailScan {
  mail_dir: string | null;
  message_count: number;
}

export interface ImportResult {
  imported: number;
  skipped: number;
  errors: number;
}

export interface ImportProgress {
  account_id: string;
  total: number;
  current: number;
  imported: number;
  skipped: number;
  done: boolean;
}

export const IMPORT_PROGRESS_EVENT = "import:progress";

export const api = {
  greet: (name: string) => invoke<string>("greet", { name }),

  listAccounts: () => invoke<AccountSummary[]>("list_accounts"),

  addAccount: (req: AddAccountRequest) =>
    invoke<AccountSummary>("add_account", { req }),

  removeAccount: (accountId: string) =>
    invoke<void>("remove_account", { accountId }),

  syncStatus: (accountId: string | null = null) =>
    invoke<SyncStatus>("sync_status", { accountId }),

  syncNow: (
    accountId: string,
    fullResync: boolean = false,
    window: SyncWindow = "month",
  ) => invoke<SyncStatus>("sync_now", { accountId, fullResync, window }),

  modelStatus: () => invoke<ModelStatus>("model_status"),

  search: (query: string, limit: number = 20) =>
    invoke<SearchResponse>("search_messages", { query, limit }),

  ask: (question: string, limit: number = 8) =>
    invoke<AnswerResponse>("ask_question", { question, limit }),

  ingestFixture: (req: IngestFixtureRequest) =>
    invoke<IngestFixtureResponse>("ingest_fixture", { req }),

  scanAppleMail: () => invoke<AppleMailScan>("scan_apple_mail"),

  importAppleMail: (accountId: string) =>
    invoke<ImportResult>("import_apple_mail", { accountId }),

  totalMessages: () => invoke<number>("total_messages"),

  syncCooldownUntil: (accountId: string) =>
    invoke<number>("sync_cooldown_until", { accountId }),
};
