# MailFind — Vision

## One-line pitch

**MailFind: unified semantic search and Q&A across all your email accounts, runs entirely on your laptop. No cloud. No subscription. Pay once.**

## The problem

Email search is broken in two ways that compound:

1. **Existing email clients have bad search.** Apple Mail's search has been functionally broken across multiple macOS versions; one Apple Discussions thread alone has 890+ "Me Too" responses spanning two years. Users describe it as "impossible," "ready to ditch," and have downloaded other apps "just to search email." This isn't a niche complaint — it's sustained, articulate, mass-scale frustration.

2. **Nobody unifies search across accounts.** Most serious email users have multiple accounts: personal, work, side-project, legacy. macOS Mail shows them all in one inbox view but search is per-account, dumb, and broken. Apple Intelligence will only ever work on iCloud. Microsoft Copilot only on Outlook. There is no tool that searches *every* account a user owns with intelligence.

The result: people lose time hunting for emails they know exist, give up, or maintain shitty workarounds (forwarding important mail to themselves, screenshot folders, separate apps per account).

## The product

A desktop app that:

- Connects to every email account you own (iCloud, Fastmail, Proton, Yahoo, generic IMAP)
- Syncs and indexes all of it locally
- Provides instant keyword search (FTS5) and semantic search (embeddings)
- Answers questions about your email with a local LLM (RAG)
- Lives alongside your existing email client — does not replace it
- Runs entirely on your machine — no cloud, no telemetry, no account
- Sold as a one-time purchase

The user keeps using Apple Mail, Outlook, Thunderbird, or whatever they already use. MailFind is the *brain* on top of their email — a search and Q&A layer that works across everything.

## The wedge

> **macOS Mail unifies *display*. MailFind unifies *intelligence*.**

The defensible positioning is **multi-account local AI**. Apple, Microsoft, and Google will each only ever serve their own ecosystem. None of them will build a tool that searches your iCloud + Fastmail + Proton + Yahoo together. That gap is structural, not temporary.

The secondary wedge is **anti-corporate-AI sentiment**: users tired of $20/mo subscriptions, tired of OpenAI/Google reading their data, tired of cloud-only AI tools. MailFind is the antidote: pay once, runs forever, nothing leaves your machine.

## The audience

The target user is **someone who runs their own email life across multiple accounts and doesn't have an IT department doing it for them**:

- Solo workers, freelancers, consultants, designers, developers
- Small business owners on non-Microsoft/non-Google email
- Privacy-conscious users on Fastmail/Proton/iCloud
- Anyone with intentionally fragmented email identity (work + personal + side projects)
- Users in r/selfhosted, r/macapps, r/privacy, r/degoogle communities

Explicitly **not** targeted:
- Pure Gmail users (Gmail web search is fine + Gemini is integrated)
- Enterprise users on Microsoft 365 (Copilot already serves them)
- Users on Intel Macs or low-spec hardware (local AI quality suffers)

## Why now

- **Local LLMs just became viable.** Quantized 3B models that run well on consumer hardware are a 2024-2025 phenomenon. Before this, "AI email search" required cloud APIs.
- **Tauri 2.0 made cross-platform indie apps viable.** Released late 2024.
- **Anti-AI-subscription sentiment is rising.** Users are exhausted by $20/mo for every productivity tool.
- **Apple Intelligence normalized on-device AI** for non-technical users.
- **The competitive window is narrow.** NeuralMail launched August 2025. semantic-mail exists. Aomail exists. None are polished commercial products yet — but the space will be crowded within 18-24 months.

## Why not VC

This is intentionally not a venture-scale startup. Realistic capture is ~100K-200K paying customers globally over several years. At $40 one-time, that's $4M-$8M lifetime revenue at full saturation, $300K-$1M at realistic capture rates. That's a great indie business and a poor venture deal.

The goal: **replace a software engineering salary with sustainable indie revenue.** ~$100K-$150K/year. No outside funding. No employees. No board.

## What success looks like

- **Year 1:** ~$0-$30K. Building, launching, marketing, supporting first cohort.
- **Year 2:** ~$30K-$100K if iteration and marketing compound.
- **Year 3:** $100K+ ARR or the product gets retired with lessons learned.

The decision point is around month 12-18. If MailFind has product-market fit by then (steady growth, low churn, organic word-of-mouth), it becomes the founder's job. If not, it gets retired and the founder goes back to a regular job with a much stronger resume and worldview.

## What MailFind is not

- Not an email client. Apple Mail / Outlook / Thunderbird remain the user's daily inbox.
- Not a Superhuman competitor. Different category (retrieval, not workflow).
- Not a Gmail tool. Gmail users have Gemini.
- Not enterprise software. No SSO, no admin console, no procurement.
- Not free. Free tier is a trap for indie products.
- Not cloud-based. Ever. No "we'll add cloud sync later" — that breaks the pitch.
