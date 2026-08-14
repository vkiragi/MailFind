// UI zoom via ⌘+ / ⌘- / ⌘0, scaling the whole app and persisting the choice.
// Uses the CSS `zoom` property (supported in the macOS WebKit webview), which
// reflows layout rather than just transform-scaling, so text stays crisp.

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
  apply();

  window.addEventListener("keydown", (e) => {
    if (!e.metaKey && !e.ctrlKey) return;
    if (e.key === "=" || e.key === "+") zoom = clamp(zoom + STEP);
    else if (e.key === "-" || e.key === "_") zoom = clamp(zoom - STEP);
    else if (e.key === "0") zoom = 1;
    else return;
    e.preventDefault();
    localStorage.setItem(ZOOM_KEY, String(zoom));
    apply();
  });
}
