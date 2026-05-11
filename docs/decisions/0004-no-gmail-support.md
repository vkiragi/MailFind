# ADR 0004 — Do not support Gmail (in v1 or any planned version)

**Status:** Accepted

**Context:**

Gmail is the dominant email provider. Supporting it would dramatically expand TAM. However, Gmail has unique constraints that make it incompatible with MailFind's positioning.

**Decision:**

Do not support Gmail. The product explicitly targets non-Gmail users.

**Reasoning:**

- **CASA security assessment.** Google requires apps using restricted Gmail scopes (`gmail.readonly` etc.) to undergo a third-party security assessment costing $15,000-$75,000, renewed annually. This is incompatible with bootstrapped indie economics.
- **App passwords aren't a clean fallback.** Gmail technically supports IMAP with app passwords, but most users have 2FA enabled and don't know what app passwords are. UX is bad.
- **Gmail users have Gemini.** Google has integrated AI search into Gmail. The pitch "local AI search" doesn't compel a Gmail user when they already have AI search built in.
- **Privacy positioning collapses.** Users who chose Gmail have implicitly accepted Google reading their email. MailFind's privacy pitch resonates with users who *didn't* choose Gmail.
- **Wedge gets sharper, not weaker.** "MailFind is for people who use email seriously and chose something other than Gmail" is a coherent identity.

**Trade-offs accepted:**

- Lose ~50% of email users globally
- Cannot serve users who have Gmail as one of multiple accounts (they'll need to use Gmail web for that account)
- Some users will install MailFind, find Gmail isn't supported, and refund

**Re-evaluation criteria:**

Reconsider only if:
- Google removes the CASA requirement (unlikely)
- The product reaches consistent revenue that could fund the annual CASA fee (~$15K min) AND the customer demand makes it worth it
- Both conditions met
