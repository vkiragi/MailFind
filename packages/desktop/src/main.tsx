import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { setupZoom } from "./lib/zoom";
import "./index.css";

// Light by default; follow the OS appearance so macOS dark mode flips the theme.
const mq = window.matchMedia("(prefers-color-scheme: dark)");
const applyTheme = () =>
  document.documentElement.classList.toggle("dark", mq.matches);
applyTheme();
mq.addEventListener("change", applyTheme);

// ⌘+ / ⌘- / ⌘0 to zoom the whole UI (persisted).
setupZoom();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
