import { invoke } from "@tauri-apps/api/core";

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

export interface IngestFixtureRequest {
  account_id: string;
  path: string;
}

export interface IngestFixtureResponse {
  imported: number;
  skipped: number;
  errors: string[];
}

export const api = {
  greet: (name: string) => invoke<string>("greet", { name }),

  listAccounts: () => invoke<AccountSummary[]>("list_accounts"),

  addAccount: (req: AddAccountRequest) =>
    invoke<AccountSummary>("add_account", { req }),

  removeAccount: (accountId: string) =>
    invoke<void>("remove_account", { accountId }),

  syncStatus: (accountId: string | null = null) =>
    invoke<SyncStatus>("sync_status", { accountId }),

  syncNow: (accountId: string, fullResync: boolean = false) =>
    invoke<SyncStatus>("sync_now", { accountId, fullResync }),

  modelStatus: () => invoke<ModelStatus>("model_status"),

  search: (query: string, limit: number = 20) =>
    invoke<SearchResponse>("search_messages", { query, limit }),

  ask: (question: string, limit: number = 8) =>
    invoke<AnswerResponse>("ask_question", { question, limit }),

  ingestFixture: (req: IngestFixtureRequest) =>
    invoke<IngestFixtureResponse>("ingest_fixture", { req }),

  totalMessages: () => invoke<number>("total_messages"),
};
