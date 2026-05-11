# MailFind — Risk Register

Short list of things that could meaningfully kill or stunt the project, with mitigations. Ordered by likelihood × impact.

## R1 — Build-then-market trap

**Risk:** Founder spends all time building, launches to silence, concludes the market doesn't want it.

**Likelihood:** High (this is the #1 indie product killer)

**Impact:** Project failure even with good product

**Mitigation:**
- Marketing track runs in parallel with build (Phase 0 of roadmap)
- Landing page + waitlist live before v0 alpha
- 500+ email signups before public launch
- Devlog posts at milestones to build audience
- Reviewer relationships established before launch, not after

## R2 — Trust hurdle for unknown indie app touching email

**Risk:** Users won't grant a no-name app full read access to their email accounts.

**Likelihood:** High

**Impact:** Low conversion regardless of product quality

**Mitigation:**
- Open-source the client (code reading user's mail must be auditable)
- Code-sign + notarize on every platform
- Security-focused privacy policy with architecture diagram
- No telemetry by default; opt-in only with clear disclosure
- Get reviewed by trusted voices (security researchers, MacStories, ATP-tier podcasts) before mainstream push
- Honest hardware compat checker builds credibility ("this company doesn't sell to people they can't serve")

## R3 — First-sync UX kills retention

**Risk:** First sync of large mailbox takes hours, looks broken, users abandon and never return.

**Likelihood:** Medium-high

**Impact:** Direct churn at the moment of highest friction

**Mitigation:**
- Tiered sync (recent first, useful at 1% complete)
- Apple Mail mbox import for instant value
- Honest progress UI ("12,000 of 94,000, recent emails are searchable now")
- Resumable across crashes
- Onboarding sets expectations explicitly

## R4 — Apple Intelligence narrows iCloud-only TAM over 2-3 years

**Risk:** Apple expands Apple Intelligence Mail features and removes the "search is broken" pain for iCloud users on modern Macs.

**Likelihood:** High over 2-3 years

**Impact:** Shrinks the iCloud-mono-user segment but does not touch multi-account users

**Mitigation:**
- Position firmly on **multi-account** wedge, not "Apple Mail search replacement"
- Marketing copy treats iCloud as one account among many
- Don't optimize the product for iCloud-only flows
- Watch Apple's WWDC announcements; pivot messaging if needed

## R5 — Cross-platform LLM inference quality varies wildly

**Risk:** Product is great on Apple Silicon, painful on mid-range Windows, broken on CPU-only Linux. Reviews on lower-tier hardware tank the brand.

**Likelihood:** High if not addressed

**Impact:** Bad reviews from real users on real hardware spread faster than good ones

**Mitigation:**
- Strict hardware floor in marketing
- Compatibility checker refuses install on unsupported hardware
- "Premium hardware required, here's why" messaging
- Don't try to support every machine — be the Apple of indie apps about hardware requirements

## R6 — Competitor ships first

**Risk:** NeuralMail, semantic-mail, or a YC company polishes their version before MailFind reaches v1, owns the category.

**Likelihood:** Medium

**Impact:** Lose first-mover positioning, harder distribution

**Mitigation:**
- Ship v0 alpha quickly (don't perfect before testing)
- Distribution and brand often beat technical first-mover (Slack, Notion, Linear all launched into competitive markets)
- Multi-platform + multi-provider + polished install + paid product is a different position than "open source GitHub project"
- The market has room for multiple products; aim for the polished commercial slot

## R7 — Microsoft 365 / Copilot eats the work-email use case

**Risk:** Most heavy email users are at work on Microsoft 365 with Copilot included. They never become MailFind customers.

**Likelihood:** Already true

**Impact:** Halves the addressable market

**Mitigation:**
- Accept it. Don't try to compete in this segment.
- Target personal email life of those same users
- Target the "I run my own email" segment (freelancers, small business, privacy-conscious, multi-account power users)

## R8 — Embedding compute time on first launch

**Risk:** Embedding 100K emails takes 10+ hours of laptop compute. Users see the laptop fan spin for a week.

**Likelihood:** High for users with large archives

**Impact:** Bad first impression, hardware concerns

**Mitigation:**
- Embeddings happen *after* FTS5 search works (FTS5 is fast to build)
- Embed last 90 days first (covers most queries)
- Pause when on battery, resume on power
- Skip junk/marketing mail by default
- Use smaller faster model on first pass, optionally re-embed in background later

## R9 — Mobile demand becomes existential

**Risk:** Top feature request is iOS app. Without it, users churn to alternatives that work on phone.

**Likelihood:** High that demand appears; medium that it kills retention

**Impact:** Could cap growth

**Mitigation:**
- Honest in marketing: "MailFind is desktop only. Search and Q&A from your laptop."
- This filters out users who'd churn anyway
- If 12 months in, mobile demand is overwhelming, revisit with eyes open (likely architecture: encrypted index sync via user's own iCloud/Dropbox, mobile reads from that)

## R10 — IMAP provider quirks consume disproportionate dev time

**Risk:** Each provider has weird edge cases (iCloud throttling, Outlook OAuth, Proton Bridge restarting, Yahoo's older IMAP server). Bug surface multiplies per provider.

**Likelihood:** High

**Impact:** Slows velocity, support load

**Mitigation:**
- Add providers sequentially, not in parallel
- Telemetry (opt-in) on connection failures by provider
- Per-provider error messages so users can self-diagnose
- "We've detected your provider is having trouble — try X" UX
- Maintain a per-provider quirks doc internally

## R11 — Founder burnout from multi-year solo grind

**Risk:** 12-18 months of building + supporting + marketing alone is exhausting. Many indie devs quit at month 8-12.

**Likelihood:** High (industry pattern)

**Impact:** Project failure regardless of metrics

**Mitigation:**
- Realistic financial runway: don't quit day job until 6+ months of revenue trajectory
- Public building creates external accountability
- Find peers in indie communities (Indie Hackers, ATP listeners, indie-focused Discords)
- Take real time off; don't grind 7 days/week
- Have a clear stop condition ("if month 12 and <$1K MRR, retire it")

## R12 — Legal / compliance surprises

**Risk:** GDPR, CCPA, EU AI Act, sales tax in 30 jurisdictions, copyright issues with model weights, app store policies. Indie devs routinely get blindsided.

**Likelihood:** Medium

**Impact:** Could force product changes, fines, distribution changes

**Mitigation:**
- Use Lemon Squeezy or Paddle (handles VAT/sales tax globally as merchant of record)
- Real privacy policy reviewed by legal services (not boilerplate)
- License terms for embedded model weights (granite is Apache 2.0, llama.cpp is MIT — both fine; verify any model swap)
- EU AI Act compliance: MailFind is "limited risk" — disclose AI use clearly, no special licensing required, but watch for changes
- Don't ship in jurisdictions you can't support (geo-block if needed)

## R13 — Conversion economics don't work

**Risk:** Trial-to-paid conversion is 2% instead of expected 10%, or refund rate is 30% instead of 5%. Math collapses.

**Likelihood:** Unknown until launch

**Impact:** Determines viability

**Mitigation:**
- Charge from day one (free tier is a trap)
- 14-day trial, not 30 (longer trials don't convert better, just delay decisions)
- Track conversion funnel from day 1 of v1
- Test pricing variants if early conversion is bad ($30 vs $40 vs $50)
- Be willing to retire if metrics never improve

## What's NOT in this risk register

These were considered and dismissed:

- "What if AI gets too smart and replaces email entirely" — speculative, not actionable
- "What if Tauri gets discontinued" — unlikely, healthy project, fallback to Electron is doable
- "What if SQLite stops being maintained" — laughable
- "What if Apple bans the app from notarization" — Apple notarizes things like this routinely
- "What if Anthropic/OpenAI release a free local model that beats granite" — that would *help* MailFind, not hurt it
