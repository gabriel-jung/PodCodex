import path from "path";
import { defineConfig } from "vitest/config";

/**
 * Separate from `vite.config.ts` on purpose. Vitest bundles its own copy of
 * Vite, so importing its `defineConfig` into the app config makes `tsc -b`
 * compare two different Vite plugin types and fail the build. The test run
 * needs none of the app config anyway: no plugins, no dev proxy, just the
 * `@/` alias.
 *
 * Node environment, with `src/test/setup.ts` supplying the two globals the
 * unit tests reach for; see that file for why a DOM environment is not the
 * answer here.
 */
export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  test: {
    include: ["src/**/*.test.ts"],
    setupFiles: ["src/test/setup.ts"],
  },
});
