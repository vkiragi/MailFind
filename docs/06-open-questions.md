# MailFind — Open Questions

Things that are NOT decided yet, that need either user research or product decisions before later phases. Captured here so they don't get forgotten or silently decided.

## Validated by talking to users (NOT by more thinking)

### Q1 — Will users install a *sidecar* app, or do they want a full client replacement?

The whole product depends on users accepting "install MailFind alongside Apple Mail/Outlook." If users actually want to *switch* email clients, MailFind is the wrong shape.

**How to validate:** During v0 alpha, watch behavior. Do users keep using their existing client and pop MailFind for search, or do they expect MailFind to *be* their email client and request reply/compose features?

**Decision deadline:** Before v1 launch. If signal says users want a full client, the product needs to fundamentally change.

### Q2 — Is "AI Q&A" the headline, or is "search that just works" the headline?

The 890+ "Me Too" thread complains about *keyword search being broken*, not "I want AI." Users might want a fast, accurate search bar, not a chatbot.

**How to validate:** During v0 alpha, log what users type. Are they typing keywords ("contractor quote") or natural-language questions ("what did the contractor quote")? Ratio matters for marketing copy.

**Decision deadline:** Before v1 marketing site is finalized.

### Q3 — What's the right pricing model?

$40 one-time is the leading hypothesis. But:
- Maybe the audience prefers $5/mo (lower barrier, you get recurring)
- Maybe the audience prefers $60-80 one-time (positions as premium)
- Maybe a free + paid tier works (despite earlier reasoning that free is a trap)

**How to validate:** Don't A/B test pricing on a small launch (statistically meaningless). Instead, ask the alpha cohort directly: "If this were $X, would you buy it? What would make you not buy?" Use answers to set v1 pricing.

**Decision deadline:** Before payment processing goes live in v1.

### Q4 — Mobile companion: when, if ever?

Top feature request post-launch will likely be iOS. Need an answer ready.

**Options:**
- "Desktop only forever" — clean, simple, limits TAM
- "Sync index via user's iCloud/Dropbox, mobile reads it" — works but breaks pure-local pitch
- "Mobile-first redesign" — different product

**How to validate:** Count post-launch requests. If iOS is in top 3 requests for 6 months, revisit. Until then, the answer is "no."

**Decision deadline:** None — this stays open indefinitely until evidence demands action.

## Decisions to make based on data

### Q5 — Should the client be open-source?

Arguments for: trust, security review, contributions, no vendor-lock-in fears.

Arguments against: someone forks it, removes the license check, distributes for free.

**Likely answer:** Open-source the client (the binary handling user mail), keep the license server / payment integration / brand closed. License key check can still be removed by a determined fork, but most paying users won't bother. Mimestream is closed-source and survives; many indie tools are open-source and survive. Both work.

**Decision deadline:** Before v1 launch (license model needs to be clear in ToS).

### Q6 — Embed or download model on first launch?

- **Embed in installer:** larger download (3-4GB total), works offline immediately
- **Download on first launch:** smaller installer (~30MB), requires internet for setup, smoother future model updates

**Likely answer:** Download on first launch. Smaller installer reduces abandonment, model updates are easier, internet is reasonable to require for initial setup. Show a clear download progress UI with size estimate.

**Decision deadline:** v0 build.

### Q7 — Which chat model is the production default?

Currently using granite4.1:3b. Alternatives: qwen2.5-3b, llama3.2-3b, phi-3.5-mini, gemma2-2b. Trade-offs: speed vs. answer quality vs. license vs. size.

**How to decide:** Benchmark on real email Q&A tasks during v0. Pick the one with the best answer quality at acceptable speed on the median target machine (M2 MacBook Air with 16GB).

**Decision deadline:** Before v1 launch.

### Q8 — How aggressive about hardware exclusion?

Strict (Apple Silicon M1+ only, no Intel ever) is cleanest but excludes meaningful audience.

Lenient (Intel Macs supported with degraded experience) is more inclusive but invites bad reviews.

**Likely answer:** Strict. Better to lose 10% of TAM and have 90% love the product than serve everyone with a mediocre experience. Apple sets this precedent.

**Decision deadline:** v0 build (compat checker logic).

## Strategic / business questions

### Q9 — Stay solo, or bring on a co-founder/designer?

UI quality is identified as the single hardest non-engineering challenge. A designer co-founder would be transformative. Also dilutes equity and complicates decisions.

**Likely answer:** Stay solo through v1. If product-market fit emerges in months 6-12, consider hiring contract designer for v1.5 polish pass. Co-founder is a high-stakes decision; defer until there's something real to share.

**Decision deadline:** Open. Revisit if v1 launch shows clear PMF and design becomes the bottleneck.

### Q10 — When (if ever) to incorporate?

LLC vs. sole proprietor vs. C-corp. Tax implications. Liability.

**Likely answer:** Sole proprietor / DBA until first revenue. LLC once revenue is steady and there's actually liability worth shielding (~6 months of revenue). C-corp only if raising money — not planned.

**Decision deadline:** When monthly revenue is consistent enough to justify the $500-1000/yr cost.

### Q11 — What's the explicit "kill it" criterion?

To avoid the indie founder trap of perpetually grinding on a project that isn't working, define a clear stop condition.

**Proposal:** If at month 18 post-launch, MRR is below $2K and growth rate is below 5%/month, retire the product. Document lessons. Move on.

**Decision deadline:** Lock this in writing before v1 launch. The point of writing it now is so future-you can't rationalize away from it.

## Things deliberately NOT open questions

These are *decided* and re-litigation is forbidden unless very strong new evidence appears:

- **Local-only architecture.** Non-negotiable. The whole pitch.
- **No Gmail.** CASA fees are prohibitive, Gemini owns the user.
- **One-time pricing.** Subscriptions don't fit local-only no-cloud product.
- **No free tier.** Free tier is an indie product killer.
- **Read-only (no sending).** Scope creep otherwise.
- **Sidecar, not replacement.** Wedge + technical scope both depend on this.

If you find yourself wanting to revisit one of these, flag it explicitly and require strong evidence — not just "wouldn't it be nice if..."
