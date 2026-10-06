import { defineConfig, devices } from "@playwright/test";

const runtimeProcess = (globalThis as typeof globalThis & { process?: { env?: Record<string, string | undefined> } }).process;
const isCi = Boolean(runtimeProcess?.env?.CI);

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  retries: isCi ? 2 : 0,
  reporter: isCi ? "github" : "list",
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "on-first-retry",
  },
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !isCi,
    timeout: 120_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
