import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AppProviders } from "./components/AppProviders";
import "./index.css";

const stored = localStorage.getItem("brain-theme");
const prefersDark =
  window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true;
document.documentElement.classList.toggle(
  "dark",
  stored ? stored === "dark" : prefersDark,
);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppProviders>
        <App />
      </AppProviders>
    </BrowserRouter>
  </React.StrictMode>,
);
