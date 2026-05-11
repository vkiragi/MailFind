# ADR 0001 — Use Tauri 2 instead of Electron

**Status:** Accepted

**Context:**

MailFind needs to be a cross-platform desktop app (macOS, Windows, Linux) with a web-tech UI and a native backend for IMAP, SQLite, and local LLM inference. The two realistic options are Electron and Tauri.

**Decision:**

Use Tauri 2 with Rust for the backend and React for the frontend.

**Reasoning:**

- **Bundle size:** Tauri ships ~10-20MB binaries vs Electron's 100MB+ baseline. For a paid indie product where users will inspect the download size, this matters.
- **Memory footprint:** Tauri uses the OS native webview, not a bundled Chromium. App is dramatically lighter at runtime.
- **Rust backend:** Memory safety for code that handles user mail. Strong async ecosystem (tokio). Cleaner async story than Electron's main process.
- **Native integration:** Better keychain access, system tray, global hotkeys, sidecars.
- **Already in use:** existing repo is built on Tauri. Switching costs would be huge for no real gain.

**Trade-offs accepted:**

- Smaller community than Electron
- WebView2 quirks on Windows
- Some Linux distributions need WebKitGTK installed
- Slightly less mature plugin ecosystem

These are acceptable. Tauri 2 is production-ready and the bundle/memory wins are significant for the product's positioning ("lightweight native app, not a fat Electron thing").
