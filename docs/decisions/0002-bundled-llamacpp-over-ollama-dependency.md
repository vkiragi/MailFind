# ADR 0002 — Bundle llama.cpp as a sidecar instead of requiring Ollama

**Status:** Accepted

**Context:**

The current README requires users to install Ollama separately and `ollama pull` two models before MailFind works. This is fine for a developer-facing alpha but unacceptable for a paid product targeting non-technical users.

**Decision:**

Bundle `llama.cpp` (or equivalent inference runtime) as a Tauri sidecar binary. Models are downloaded on first launch by MailFind itself, not by the user via Ollama.

**Reasoning:**

- **One-click install.** User experience must be: download .dmg → drag to Applications → open → onboarding handles everything. "Go install Ollama and run these commands" is a non-starter for the target audience.
- **No external dependency drift.** If Ollama updates and breaks API compatibility, MailFind doesn't break.
- **Hardware tuning.** We can pick build flags optimized for our use case (Metal on Apple Silicon, CUDA on NVIDIA, etc.).
- **License clarity.** llama.cpp is MIT-licensed; bundling is straightforward.

**Trade-offs accepted:**

- We maintain our own inference runtime build pipeline per platform
- Users with Ollama already installed can't reuse it (could expose this as advanced setting later)
- We're responsible for security updates to the inference runtime

**Future consideration:**

In v2 or later, we might expose an "advanced: use my existing Ollama install" option for power users. Not v1.
