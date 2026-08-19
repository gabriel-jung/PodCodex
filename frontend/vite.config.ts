import { readFileSync } from "node:fs";
import os from "node:os";
import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev-only: authenticate the browser against the loopback API by injecting
// the auth header at the proxy, server-side. Reads the `api_token` file the
// FastAPI server creates (path mirrors src/podcodex/core/api_token.py;
// sync-checked by tests/test_frontend_constants_sync.py). Read-only and
// lazy: if the backend hasn't booted yet, requests 401 until it has, then
// the next request picks the token up. No token ever reaches the browser
// or the built bundle.
function readApiToken(): string {
  const env = process.env.PODCODEX_API_TOKEN;
  if (env) return env;
  const base = process.env.XDG_CONFIG_HOME
    ? path.join(process.env.XDG_CONFIG_HOME, "podcodex")
    : path.join(os.homedir(), ".config", "podcodex");
  try {
    return readFileSync(path.join(base, "api_token"), "utf-8").trim();
  } catch {
    return "";
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:18811",
        changeOrigin: true,
        ws: true,
        // Rewrite the host in 30x `Location` headers back to the dev server.
        // FastAPI's trailing-slash redirect points at the target absolutely,
        // and `changeOrigin` makes that `127.0.0.1:18811` — the browser would
        // follow it straight to the backend, escaping this proxy and the
        // token it injects, and land on 401 with no visible error.
        autoRewrite: true,
        configure: (proxy) => {
          // Read per request (tiny local file, dev only) so a token that
          // appears or changes after Vite boots is picked up without a
          // restart.
          const inject = (proxyReq: { setHeader(k: string, v: string): void }) => {
            const token = readApiToken();
            if (token) proxyReq.setHeader("X-PodCodex-Token", token);
          };
          proxy.on("proxyReq", inject);
          proxy.on("proxyReqWs", inject);
        },
      },
    },
  },
});
