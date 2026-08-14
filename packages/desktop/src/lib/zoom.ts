// UI zoom, driven by the native View menu (Zoom In / Out / Actual Size) and
// their ⌘ shortcuts, plus a direct ⌘= fallback. Uses the CSS `zoom` property
// (supported in the macOS WebKit webview), which reflows layout rather than
// transform-scaling, so text stays crisp. The choice is persisted.

import { listen } from "@tauri-apps/api/event";

const ZOOM_KEY = "mf:zoom";
const MIN = 0.8;
const MAX = 1.6;
const STEP = 0.1;

const clamp = (z: number) => Math.min(MAX, Math.max(MIN, Math.round(z * 10) / 10));

export function setupZoom() {
  let zoom = clamp(parseFloat(localStorage.getItem(ZOOM_KEY) || "1") || 1);
  const apply = () => {
    // `zoom` isn't in the typed CSSStyleDeclaration; set it as a property.
    document.documentElement.style.setProperty("zoom", String(zoom));
  };
  const set = (z: number) => {
    zoom = clamp(z);
    localStorage.setItem(ZOOM_KEY, String(zoom));
    apply();
  };
  apply();

  // Native View-menu items emit this; the menu owns the ⌘+/⌘-/⌘0 accelerators.
  listen<string>("menu://zoom", (e) => {
    if (e.payload === "zoom_in") set(zoom + STEP);
    else if (e.payload === "zoom_out") set(zoom - STEP);
    else if (e.payload === "zoom_reset") set(1);
  }).catch(() => {});

  // Fallback for plain ⌘= (the menu accelerator binds ⌘+, i.e. shift+=).
  window.addEventListener("keydown", (e) => {
    if (!e.metaKey && !e.ctrlKey) return;
    if (e.key === "=") set(zoom + STEP);
    else return;
    e.preventDefault();
  });
}
