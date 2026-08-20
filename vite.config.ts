import { defineConfig } from "vite";

// The engine core (src/core) is framework- and DOM-free so it runs identically
// under Node (validation harness, export) and in the browser Web Worker.
// Vite only bundles the thin live-viz app in src/app + the worker.
export default defineConfig({
  root: ".",
  build: {
    target: "es2022",
    outDir: "dist",
  },
  worker: {
    format: "es",
  },
});
