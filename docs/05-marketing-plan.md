# MailFind — Marketing Plan

Marketing for an indie product is not a phase. It runs continuously, parallel to development, from now until product retirement. This document defines the work.

## Positioning

**Primary positioning (the headline):**
> Search every email account you own with AI. Locally. No cloud. No subscription. Pay once.

**Secondary positioning (for technical/privacy audiences):**
> Your email never leaves your machine. Neither does the AI.

**Tertiary positioning (for the frustrated Apple Mail user):**
> Apple Mail search is broken. We fixed it — across every account you own.

Use the right one for the right audience. Never use all three at once on the same surface.

## Audience segments and where they live

| Segment | Where | Approach |
|---|---|---|
| Privacy nerds | /r/privacy, /r/selfhosted, /r/degoogle, Hacker News, Lobsters | Lead with "no cloud, runs locally, open-source client" |
| Mac power users | /r/macapps, MacRumors forums, MacStories audience, ATP listeners, MPU forum | Lead with "fixes Apple Mail search across all accounts" |
| Multi-account email users | Fastmail subreddit, Proton subreddit, indie hackers | Lead with "unified search across iCloud, Fastmail, Proton, Yahoo" |
| Anti-AI-subscription crowd | HN comments on AI pricing posts, indie newsletters | Lead with "pay once, runs forever, no OpenAI subscription" |

Different copy per surface. Same product.

## Channel strategy

### Owned channels (build these first)

- **Landing page** at mailfind.app (or similar). Above-the-fold pitch + screenshots + demo video + waitlist signup. Live by end of Phase 0.
- **Email waitlist.** Ferry signups via ConvertKit or Buttondown. Use it for launch announcement, major updates, version 1 release.
- **Twitter/X account.** Build-in-public devlogs, screenshots, milestones. Aim for 1-3 posts/week.
- **Personal blog or dev.to.** Long-form devlogs at major milestones (5-10 posts over the v0+v1 build). These get found via Google months later.
- **GitHub repo (open-source client).** README itself is marketing. Pinned issues for roadmap visibility. Open issues for community trust.

### Earned channels (cultivate during build, activate at launch)

- **Hacker News Show HN.** The big one. One shot. See "Launch playbook" below.
- **Product Hunt launch.** Lower-impact than HN for technical products but worth doing.
- **MacStories / Six Colors / The Verge / The Sweet Setup.** Email the writers individually, not press@. Personal message, early access, screenshots.
- **Podcasts:** ATP, Mac Power Users, Connected, Cortex. Listeners are ideal customers. Email hosts with a personal pitch and free license.
- **Newsletters:** Indie Mac Apps, Indie Hackers, Hacker Newsletter. Pitch them when launched.
- **YouTube reviewers:** Snazzy Labs, Jonathan Morrison, MKBHD-tier (less likely but try). For v2 with Windows: Linus Tech Tips audience.

### Community channels (continuous low-touch presence)

- /r/macapps — answer questions, mention MailFind only when relevant
- /r/selfhosted — same
- /r/privacy — same
- Hacker News commenting — be a useful presence in email/AI/privacy threads
- Indie Hackers forum — post milestone updates
- Apple Discussions threads about Mail search — *carefully* mention MailFind once it's live (don't spam)

## Pre-launch waitlist building

Goal: **500+ email signups before public launch.**

How to get there:

1. **Apple Discussions thread outreach.** The 890+ "Me Too" thread alone has hundreds of frustrated users. Personal messages — not spam — saying "I'm building a fix for this, want to be notified?" Likely yields 30-50 signups from one thread.
2. **Reddit posts.** Not "buy my product" posts — "I'm building X, what would you want from it?" posts. These get upvoted and signups when authentic.
3. **Twitter build-in-public.** Tag #buildinpublic, screenshot Tuesdays, post UI iterations. Slow but compounding.
4. **Friend/network outreach.** Personal email to anyone you know who uses non-Gmail email.
5. **Devlog posts.** Long-form posts on personal blog or dev.to. SEO compounds.
6. **Podcast guest appearances.** Pitch "the building of MailFind" to indie-focused podcasts.

## Launch playbook (the Show HN moment)

Show HN is the single highest-leverage marketing event. Treat it accordingly.

**Pre-launch (week before):**

- Final draft of Show HN post, read by 2-3 trusted reviewers
- Demo video (90 seconds, no music, just the product working) embedded in landing page
- Pricing finalized, payment processing live and tested with real cards
- Refund policy live
- Privacy policy live
- All known bugs from beta fixed
- Code-signed installer downloadable from landing page
- License key flow tested end-to-end
- Server/CDN can handle ~10K visitors in an hour (Vercel handles this)
- Email to waitlist drafted

**Launch day (Tuesday morning, ~7-9 AM Pacific):**

- Show HN post goes up
- Send waitlist email simultaneously
- Tweet thread with the Show HN link
- Post in /r/macapps (read their rules first)
- Post in /r/selfhosted, /r/privacy, /r/Mac
- Email reviewers you've been in touch with: "We're live, if you want to cover this is the day"
- Stand by to respond to *every* HN comment within 30 minutes for the first 6 hours

**Show HN post structure:**

```
Title: Show HN: MailFind – Local AI search across all your email accounts

Body:
Hi HN! I'm [name], and I built MailFind because Apple Mail search has been broken for years and no tool searches across multiple email accounts with AI.

What it does: [2-3 sentences]

How it works: [2-3 sentences on the local-only architecture]

Why I built it: [personal story, ~100 words]

Tech stack: Tauri + Rust + SQLite + llama.cpp. Open-source client, paid binary.

Pricing: $40 one-time, 14-day trial.

Hardware requirements: [be explicit]

What it doesn't do: [Gmail, mobile, etc. — set expectations]

Happy to answer questions.
```

The post must be honest, specific, and humble. HN sniffs marketing bullshit instantly.

**Post-launch (week after):**

- Reply to every comment, every email, every tweet
- Fix any urgent bugs reported and ship a patch
- Capture testimonials from happy launch-day customers
- Write a "lessons from launch" post the following week (more SEO content)

## Pricing communication

The price needs to be the third thing on the landing page (after pitch and screenshot), not buried.

**Recommended structure:**

```
$40 — One-time purchase
14-day free trial. No subscription. No account. Lifetime updates for current major version.

[Download trial]
[Buy now]
```

The "no subscription" line is the killer feature for the anti-AI-subscription crowd. Make it loud.

## Sustained marketing (months 3-12)

After launch, the slow grind:

- 1 longform post per month (devlog, technical deep-dive, case study)
- 1-2 social posts per week
- Respond to forum mentions of "Apple Mail search broken" with a soft mention of MailFind
- Reach out to 1-2 reviewers per month who haven't covered it yet
- Quarterly product updates with email blast to customers + waitlist
- Annual "year in review" post

## What NOT to do

- **No paid ads at launch.** Indie products with niche audiences don't get ROI from Google/Facebook ads. Skip unless something specific changes.
- **No SEO content farms.** Don't write 50 thin blog posts targeting keywords. Write 5 great ones.
- **No influencer marketing.** Indie tech audiences hate it.
- **No fake reviews or astroturfing.** HN catches it instantly. Fatal to the brand.
- **No comparison hit pieces.** "Why MailFind beats Mimestream" is bad form. Mimestream isn't your competitor — Apple Mail is.
- **No premature scaling.** Don't set up customer support tools, status pages, or community Discord before you have customers asking for them.

## Metrics to track

The minimum viable analytics:

- Landing page visitors (Plausible, not Google Analytics — privacy crowd hates GA)
- Conversion to email signup
- Trial downloads
- Trial-to-paid conversion
- Refund rate
- Customer acquisition channel (ask in onboarding: "How did you hear about MailFind?")

Don't measure what you can't act on. Don't add 20 SaaS analytics tools. The 6 numbers above are enough.

## Brand voice

- **Honest** — about hardware requirements, about what doesn't work, about being indie
- **Technical** — assume the reader knows what IMAP is
- **Anti-corporate** — gentle, not bitter; the audience is here for a reason
- **Not overly clever** — no AI jokes, no emoji, no "we're disrupting email"
- **Confident, not boastful** — "this works" not "this is the best"

The voice that works for this audience: imagine writing for Hacker News, but a kinder version. Direct, honest, specific.
