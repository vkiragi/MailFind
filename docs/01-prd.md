# MailFind — Product Requirements

This document defines what gets built, scoped per phase. Each phase has explicit completion criteria. Ship when criteria are met.

## Phase scope summary

| Phase | Audience | Providers | Distribution | Pricing |
|---|---|---|---|---|
| **v0 — Private alpha** | 5-10 invited testers | iCloud only | Direct download, signed | Free |
| **v1 — Public launch** | Public | iCloud, Fastmail, generic IMAP | Public site, Show HN | $40 one-time |
| **v1.1 — Provider expansion** | Public | + Proton (via Bridge), Yahoo | Public site | $40 one-time |
| **v2 — Cross-platform** | Public | All v1.1 providers | Public site | $40 one-time |

## v0 — Private alpha

**Goal:** validate that real users (a) successfully install and use it, (b) find it useful, (c) trust it enough to give feedback, (d) reveal which assumptions in this plan are wrong.

### Required features

- **iCloud Mail sync via IMAP + app password.** Recent-first sync (last 30 days first), then progressively older. Resumable across crashes/restarts.
- **Apple Mail import.** On first launch, offer to import existing `.mbox` data from `~/Library/Mail/`. Instant value, no IMAP wait.
- **Local SQLite storage** with full message body, headers, metadata, account ID.
- **FTS5 keyword search** across all synced mail. Sub-200ms responses.
- **Hybrid search** combining FTS5 + vector embeddings (nomic-embed-text via Ollama).
- **RAG Q&A** with citations. User asks a question, sees a streamed answer with clickable links to source emails.
- **Global hotkey overlay** (Cmd+Shift+E or similar). Press from anywhere, search/ask, dismiss.
- **Menu bar icon** showing sync status and unread counts.
- **Hardware compatibility check** on first launch. Refuse to install on Intel Macs / <16GB RAM with clear explanation.
- **Code-signed and notarized** macOS installer (.dmg).
- **Threading** via JWZ algorithm. Group thread messages in results.
- **Quote-stripping** before embedding (use email-reply-parser patterns).
- **Crash reporting** (opt-in, anonymous) so you can debug user issues you can't reproduce.

### Explicitly excluded from v0

- Other email providers (iCloud only)
- Windows/Linux support
- Payments (alpha is free)
- Public marketing site
- Any cloud component
- iOS/mobile companion
- Settings UI beyond bare minimum
- Custom embedding models (use nomic-embed-text only)
- Attachment text extraction

### v0 completion criteria

- [ ] 5-10 alpha users successfully installed and synced their iCloud Mail
- [ ] At least 3 users have used the search/Q&A daily for a week
- [ ] At least 2 users have explicitly said they would pay for the public version
- [ ] Major bugs from alpha logs are fixed
- [ ] First-sync UX doesn't make users abandon the app
- [ ] Q&A returns useful answers (not hallucinated, properly cited) on real user mailboxes

## v1 — Public launch

**Goal:** real public release with paying customers. The Show HN moment.

### Added features over v0

- **Fastmail support** (IMAP + app password). Same flow as iCloud.
- **Generic IMAP support** (manual server config). Covers self-hosted, Posteo, Mailbox.org, Zoho, university servers, etc.
- **Multi-account UI.** Add/remove accounts in settings. Per-account sync status. Search results tagged by account.
- **Cross-account dedup** (Message-ID hashing).
- **Onboarding flow.** Compatibility check → choose providers → sign in → import from Apple Mail (optional) → "your first sync is starting, here's what to expect."
- **Settings UI.** Account management, sync preferences, model selection (default + smaller fallback), hotkey customization.
- **Trial mechanism.** 14-day free trial, then $40 one-time purchase.
- **Payment integration** via Lemon Squeezy or Paddle (handles tax/VAT globally for indie devs).
- **License key system.** Customer pays → gets license key → enters in app → unlocks beyond trial.
- **Update mechanism.** App can self-update from a release feed without losing user data.
- **Better error states.** Per-provider error messages ("iCloud is throttling, retrying in 60s") instead of silent failures.
- **"Search the raw index without AI" fallback.** When the LLM is wrong, user can verify the email exists.
- **Privacy policy + ToS.** Real ones, written by you (not AI-generated, not boilerplate).

### Required infrastructure

- Marketing site (Vercel deploy, email capture, screenshots, pricing, FAQ)
- Compatibility checker tool on the site (downloadable utility that tests user's hardware)
- Lemon Squeezy storefront live and tested
- Privacy policy that's actually accurate
- Refund policy (30-day, no questions)
- Code signing for both Mac (Apple Developer Program) — Windows/Linux deferred to v2
- A real domain, real email address for support
- Beta testimonials usable in marketing copy
- Show HN post drafted and revised

### v1 completion criteria

- [ ] All v0 criteria still met
- [ ] All three providers (iCloud, Fastmail, generic IMAP) tested with real accounts
- [ ] Onboarding flow tested with someone who has never seen the app before
- [ ] Trial → purchase → license key flow works end-to-end
- [ ] Marketing site live and converts visitors to email signups at >15%
- [ ] Privacy policy reviewed (by a lawyer or legal-services tool, not just self-written)
- [ ] At least 100 emails in the pre-launch waitlist
- [ ] Show HN post drafted and read by 2-3 people for feedback

## v1.1 — Provider expansion

**Goal:** broaden the addressable user base by adding the two largest non-Gmail providers still missing.

### Added features

- **Proton Mail support** via Proton Bridge. Onboarding instructions for installing Bridge. Detect Bridge running, auto-configure connection.
- **Yahoo Mail support** (IMAP + app password).
- **Improved attachment handling** — extract text from PDFs and Word docs for indexing.
- **Bulk archive sync mode** — for users with very large mailboxes (100K+ emails), better progress UI and pause/resume controls.

### v1.1 completion criteria

- [ ] Proton + Yahoo tested with real accounts
- [ ] Proton Bridge onboarding doesn't lose users (test with someone who hasn't installed Bridge)
- [ ] Attachment text appears in search results

## v2 — Cross-platform

**Goal:** Windows and Linux support, expanding addressable market beyond Mac.

### Added features

- **Windows installer** (MSI + EV cert).
- **Linux packages** (.deb, .rpm, AppImage).
- **Per-platform keychain integration** (Windows Credential Manager, GNOME Keyring/KWallet).
- **Per-platform LLM inference** (CUDA on Windows with NVIDIA, Vulkan/ROCm fallback, CPU-only honest about being slow).
- **Microsoft Outlook OAuth support** (Microsoft 365, Hotmail) — required because Microsoft is deprecating basic auth.

### v2 completion criteria

- [ ] All providers from v1.1 work on Windows and Linux
- [ ] Outlook OAuth flow approved by Microsoft
- [ ] Hardware compat checker covers all three platforms
- [ ] At least 10 users on Windows and 10 on Linux successfully running the app

## Future consideration (not committed)

- Mobile companion (iOS/Android) — needs separate architectural decision; default answer for foreseeable future is "desktop only"
- Cloud sync of encrypted index across user's own devices (would unlock mobile but breaks pure-local pitch)
- Calendar/Notes/Slack indexing — expansion beyond email
- Premium tier with cloud-optional features
- Team/Family licensing

## Out of scope, ever

- Sending email (read-only product)
- Replacing the user's email client
- Storing user data on our servers
- Free tier
- Telemetry by default
- Gmail support (CASA fees, Gemini owns that user)
- Encrypted-only providers without IMAP (Tutanota, etc.)
