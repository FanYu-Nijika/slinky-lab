import { test, expect, type Page } from "@playwright/test";

const config = {
  schema_version: 1,
  name: "悬挂下落",
  scenario: "drop",
  material: { turns: 12, radius: 0.03, strip_width: 0.003, strip_thickness: 0.0015, pitch: 0.0017, mass: 0.0487, young_modulus: 100000000, shear_modulus: 37000000, damping: 0.00001, friction: 0.5 },
  scene: { gravity: 9.81, step_height: 0.04, step_depth: 0.08, step_width: 0.3, step_count: 6, tilt_deg: 25, initial_angular_velocity: 0, initial_forward_velocity: 0, launch_offset: 0, settle_time: 2 },
  numerics: { profile: "preview", duration: 1.5, sample_hz: 60, segments_per_turn: null, timestep: null, max_wall_seconds: 600 },
  provenance: { preset: "test" },
};

async function mockApi(page: Page) {
  await page.route("**/api/v1/presets", (route: any) => route.fulfill({ json: [{ id: "drop", label: "悬挂下落", description: "test", config }] }));
  await page.route("**/api/v1/runs", async (route: any) => {
    if (route.request().method() === "POST") return route.fulfill({ json: { run_id: "run-test", status: "queued" } });
    return route.fulfill({ json: [] });
  });
  await page.route("**/api/v1/runs/run-test", (route: any) => route.fulfill({ json: { run_id: "run-test", status: "completed", config, summary: {}, artifacts: [] } }));
  await page.route("**/api/v1/runs/run-test/frames**", (route: any) => route.fulfill({ json: { frames: [{ time: 0, positions: [[0, 0, 0.1]], quaternions: [[1, 0, 0, 0]], contacts: [], metrics: { com_z: 0.1 } }], total: 1, metadata: { half_sizes: [[0.001, 0.001, 0.001]], engine_version: "3.15.0", static_geoms: [] } } }));
  await page.route("**/api/v1/runs/run-test/commands", (route: any) => route.fulfill({ json: { run_id: "run-test", status: "paused", config, summary: {}, artifacts: [] } }));
}

test("opens the Chinese research workbench and changes geometry controls", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await expect(page.getByText("三维动力学视图")).toBeVisible();
  await expect(page.getByTestId("parameter-panel")).toBeVisible();
  await page.getByRole("button", { name: /平滑带状/ }).click();
  await expect(page.getByText("平滑形状示意")).toBeVisible();
  await page.getByRole("button", { name: /轨迹/ }).click();
  await page.getByRole("button", { name: /悬挂下落/ }).click();
  await expect(page.getByText("场景")).toBeVisible();
});

test("submits a run, shows it in history, and supports timeline/export controls", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByTestId("start-run").click();
  await expect(page.getByText("已完成").first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId("timeline")).toBeVisible();
  await expect(page.getByRole("button", { name: "JSON" })).toBeEnabled();
});

test("opens the 2D scan editor and validates scan controls", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByTestId("sweep-panel").getByRole("button", { name: /参数扫描/ }).click();
  await page.getByRole("button", { name: "2D" }).click();
  await expect(page.getByText("Y 轴")).toBeVisible();
  await expect(page.getByRole("button", { name: /开始网格扫描/ })).toBeVisible();
});
