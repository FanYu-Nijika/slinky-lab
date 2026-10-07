import { test, expect, type Page } from "@playwright/test";
import { DEFAULT_DROP_VALIDATION_CONFIG } from "../src/types";

const screenshotDirectory = "../reports";

const config: any = {
  schema_version: 1,
  name: "悬挂下落",
  scenario: "drop",
  material: { turns: 12, radius: 0.03, strip_width: 0.003, strip_thickness: 0.0015, pitch: 0.0017, mass: 0.0487, young_modulus: 100000000, shear_modulus: 37000000, damping: 0.00001, friction: 0.5 },
  scene: { gravity: 9.81, step_height: 0.04, step_depth: 0.08, step_width: 0.3, step_count: 6, tilt_deg: 25, initial_angular_velocity: 0, initial_forward_velocity: 0, launch_offset: 0, settle_time: 2 },
  numerics: { profile: "preview", duration: 1.5, sample_hz: 60, segments_per_turn: null, timestep: null, max_wall_seconds: 600 },
  provenance: { preset: "test" },
};

async function mockApi(page: Page, onRunConfig?: (runConfig: any) => void, presetList = [{ id: "drop", label: "悬挂下落", description: "test", config }]) {
  await page.route("**/api/v1/presets", (route: any) => route.fulfill({ json: presetList }));
  await page.route("**/api/v1/runs", async (route: any) => {
    if (route.request().method() === "POST") {
      onRunConfig?.(route.request().postDataJSON());
      return route.fulfill({ json: { run_id: "run-test", status: "queued" } });
    }
    return route.fulfill({ json: [] });
  });
  await page.route("**/api/v1/runs/run-test", (route: any) => route.fulfill({ json: { run_id: "run-test", status: "completed", config, summary: {}, artifacts: [] } }));
  await page.route("**/api/v1/runs/run-test/frames**", (route: any) => route.fulfill({ json: { frames: [{ time: 0, positions: [[0, 0, 0.1]], quaternions: [[1, 0, 0, 0]], contacts: [], metrics: { com_z: 0.1 } }], total: 1, metadata: { half_sizes: [[0.001, 0.001, 0.001]], engine_version: "3.15.0", static_geoms: [] } } }));
  await page.route("**/api/v1/runs/run-test/commands", (route: any) => route.fulfill({ json: { run_id: "run-test", status: "paused", config, summary: {}, artifacts: [] } }));
}

async function mockPreparingApi(page: Page, actions: string[]) {
  let currentStatus = "preparing";
  await page.route("**/api/v1/presets", (route: any) => route.fulfill({ json: [{ id: "drop", label: "悬挂下落", description: "test", config }] }));
  await page.route("**/api/v1/runs", async (route: any) => {
    if (route.request().method() === "POST") return route.fulfill({ json: { run_id: "run-preparing", status: "queued" } });
    return route.fulfill({ json: [] });
  });
  await page.route("**/api/v1/runs/run-preparing", (route: any) => route.fulfill({ json: { run_id: "run-preparing", status: currentStatus, config, summary: {}, artifacts: [] } }));
  await page.route("**/api/v1/runs/run-preparing/frames**", (route: any) => route.fulfill({ json: { frames: [], total: 0, metadata: { half_sizes: [], engine_version: "3.15.0", static_geoms: [] } } }));
  await page.route("**/api/v1/runs/run-preparing/commands", async (route: any) => {
    const action = route.request().postDataJSON().action as string;
    actions.push(action);
    currentStatus = action === "cancel" ? "cancelled" : action === "pause" ? "paused" : action === "resume" ? "preparing" : "paused";
    return route.fulfill({ json: { run_id: "run-preparing", status: currentStatus, config, summary: {}, artifacts: [] } });
  });
}

async function mockAmbiguousStairsApi(page: Page) {
  const stairsConfig = { ...config, name: "翻转下楼梯", scenario: "stairs" };
  const summary = {
    numerical_validation: "passed",
    movement_classification: "ambiguous_flip",
    support_diagnostics: {
      candidate_flip_count: 3,
      confirmed_flip_count: 0,
      events: [
        { step: 1, first_touch_label: "last", first_sustained_label: "first", endpoint_supports: [{ label: "first", time: 0.1 }], ambiguous: true, ambiguity_reasons: ["端圈支撑交替"] },
        { step: 2, first_touch_label: "first", first_sustained_label: "last", endpoint_supports: [{ label: "last", time: 0.2 }], ambiguous: true, ambiguity_reasons: ["持续支撑不足"] },
        { step: 3, first_touch_label: "last", first_sustained_label: null, endpoint_supports: [], ambiguous: true, ambiguity_reasons: ["端圈未建立"] },
      ],
    },
    step_events: [{ step: 1, time: 0.1, end: "last" }, { step: 2, time: 0.2, end: "first" }],
  };
  await page.route("**/api/v1/presets", (route: any) => route.fulfill({ json: [{ id: "stairs", label: "翻转下楼梯", description: "test", config: stairsConfig }] }));
  await page.route("**/api/v1/runs", async (route: any) => {
    if (route.request().method() === "POST") return route.fulfill({ json: { run_id: "run-ambiguous", status: "queued" } });
    return route.fulfill({ json: [] });
  });
  await page.route("**/api/v1/runs/run-ambiguous", (route: any) => route.fulfill({ json: { run_id: "run-ambiguous", status: "completed", config: stairsConfig, summary, artifacts: [] } }));
  await page.route("**/api/v1/runs/run-ambiguous/frames**", (route: any) => route.fulfill({ json: { frames: [{ time: 0, positions: [[0, 0, 0.1]], quaternions: [[1, 0, 0, 0]], contacts: [], metrics: { com_z: 0.1 } }], total: 1, metadata: { half_sizes: [[0.001, 0.001, 0.001]], engine_version: "3.15.0", static_geoms: [] } } }));
  await page.route("**/api/v1/runs/run-ambiguous/commands", (route: any) => route.fulfill({ json: { run_id: "run-ambiguous", status: "paused", config: stairsConfig, summary, artifacts: [] } }));
}

function inputFor(page: Page, label: string) {
  return page.locator("label.field").filter({ hasText: label }).locator("input");
}

test("opens the Chinese research workbench and changes geometry controls", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByLabel("后端预设").selectOption("drop");
  await expect(page.getByText("三维动力学视图")).toBeVisible();
  await expect(page.getByTestId("parameter-panel")).toBeVisible();
  await expect(page.getByRole("button", { name: /彩虹带/ })).toHaveClass(/active/);
  await expect(page.getByRole("button", { name: /跟随质心/ })).toHaveClass(/active/);
  await expect(page.getByText("彩虹带 · 插值表面 · 几何预览（未运行物理仿真）")).toBeVisible();
  await expect(page.getByTitle("取消")).toBeDisabled();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${screenshotDirectory}/ui-preview.png` });
  await page.getByRole("button", { name: /碰撞几何/ }).click();
  await expect(page.getByText("碰撞几何 · 几何预览")).toBeVisible();
  await page.getByRole("button", { name: /彩虹带/ }).click();
  await expect(page.getByText("彩虹带 · 插值表面 · 几何预览")).toBeVisible();
  await page.getByRole("button", { name: /轨迹/ }).click();
  await page.getByRole("button", { name: /悬挂下落/ }).click();
  await expect(page.getByText("场景")).toBeVisible();
});

test("submits a run, shows it in history, and supports timeline/export controls", async ({ page }) => {
  let commandCalls = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().includes("/commands")) commandCalls += 1;
  });
  await mockApi(page);
  await page.goto("/");
  await page.getByTestId("start-run").click();
  await expect(page.getByText("已完成").first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId("result-summary")).toBeVisible();
  await expect(page.getByText("数值检查")).toBeVisible();
  await expect(page.getByText("实验支持")).toBeVisible();
  await expect(page.getByTestId("timeline")).toBeVisible();
  await expect(page.getByRole("button", { name: "JSON" })).toBeEnabled();
  await expect(page.getByTestId("model-version")).toHaveText("引擎 3.15.0");
  await page.getByTitle("回放单步").click();
  await page.waitForTimeout(100);
  expect(commandCalls).toBe(0);
});

test("keeps preparation pause, step, and cancel controls actionable", async ({ page }) => {
  const actions: string[] = [];
  await mockPreparingApi(page, actions);
  await page.goto("/");
  await page.getByTestId("start-run").click();
  await expect.poll(() => actions.length).toBe(0);
  const playButton = page.getByRole("button", { name: /暂停/ });
  await expect(playButton).toBeEnabled();
  await expect(page.getByTitle("单步")).toBeDisabled();
  await expect(page.getByTitle("取消")).toBeEnabled();

  await playButton.click();
  await expect.poll(() => actions).toContain("pause");
  await expect(page.getByText("已暂停").first()).toBeVisible();
  await expect(page.getByTitle("完成准备并停在首帧")).toBeEnabled();

  await page.getByTitle("完成准备并停在首帧").click();
  await expect.poll(() => actions).toContain("step");
  await page.getByTitle("取消").click();
  await expect.poll(() => actions).toContain("cancel");
  await expect(page.getByText("已取消").first()).toBeVisible();
});

test("loads the backend drop-validation preset on first open", async ({ page }) => {
  const validationPreset = {
    id: "drop-validation",
    label: "下落数值验证小算例（3 圈）",
    description: "test",
    config: DEFAULT_DROP_VALIDATION_CONFIG,
  };
  await mockApi(page, undefined, [validationPreset]);
  await page.goto("/");
  await expect(inputFor(page, "圈数")).toHaveValue("3");
  await expect(inputFor(page, "名称")).toHaveValue("下落数值验证小算例");
  await page.getByText("展开高级数值设置", { exact: true }).click();
  await expect.poll(async () => Number(await inputFor(page, "每圈段数").inputValue())).toBe(16);
  await expect.poll(async () => Number(await inputFor(page, "积分步长").inputValue())).toBeCloseTo(0.0000125, 12);
});

test("does not replace an edit while the default preset request is pending", async ({ page }) => {
  await page.route("**/api/v1/presets", async (route: any) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.fulfill({ json: [{ id: "drop-validation", label: "下落数值验证小算例（3 圈）", description: "test", config: DEFAULT_DROP_VALIDATION_CONFIG }] });
  });
  await page.route("**/api/v1/runs", (route: any) => route.fulfill({ json: [] }));
  await page.goto("/");
  const name = inputFor(page, "名称");
  await name.fill("用户先行编辑");
  await expect(name).toHaveValue("用户先行编辑");
  await page.waitForTimeout(700);
  await expect(name).toHaveValue("用户先行编辑");
});

test("uses the local three-turn validation config when presets are offline", async ({ page }) => {
  await page.route("**/api/v1/presets", (route: any) => route.abort());
  await page.route("**/api/v1/runs", (route: any) => route.abort());
  await page.goto("/");
  await expect(inputFor(page, "圈数")).toHaveValue("3");
  await expect(inputFor(page, "名称")).toHaveValue("下落数值验证小算例");
  await page.getByText("展开高级数值设置", { exact: true }).click();
  await expect.poll(async () => Number(await inputFor(page, "每圈段数").inputValue())).toBe(16);
  await expect.poll(async () => Number(await inputFor(page, "积分步长").inputValue())).toBeCloseTo(0.0000125, 12);
});

test("submits the extended physical and stairs parameters independently", async ({ page }) => {
  let postedConfig: any;
  await mockApi(page, (runConfig) => { postedConfig = runConfig; });
  await page.goto("/");
  await page.getByRole("button", { name: /翻转下楼梯/ }).click();

  await page.locator("details.param-section").filter({ hasText: "弹性与接触" }).locator("summary").click();
  await inputFor(page, "杨氏模量 E").fill("123000000");
  await inputFor(page, "剪切模量 G").fill("45600000");
  await inputFor(page, "铰接阻尼").fill("0.000023");
  await inputFor(page, "台阶/地面摩擦").fill("0.41");
  await inputFor(page, "圈间摩擦").fill("0.73");
  await page.getByText("展开高级数值设置", { exact: true }).click();
  await inputFor(page, "接触时间常数").fill("0.0045");
  await inputFor(page, "计算时间上限").fill("720");
  await inputFor(page, "阶宽").fill("0.44");
  await inputFor(page, "初始侧向速度").fill("0.12");
  await inputFor(page, "初始角速度").fill("1.7");
  await inputFor(page, "边缘摆放偏移").fill("0.015");

  await page.getByTestId("start-run").click();
  await expect(page.getByText("已完成").first()).toBeVisible({ timeout: 10_000 });
  expect(postedConfig.material.young_modulus).toBe(123000000);
  expect(postedConfig.material.shear_modulus).toBe(45600000);
  expect(postedConfig.material.damping).toBe(0.000023);
  expect(postedConfig.material.friction).toBe(0.41);
  expect(postedConfig.material.self_friction).toBe(0.73);
  expect(postedConfig.material.friction).not.toBe(postedConfig.material.self_friction);
  expect(postedConfig.scene.step_width).toBe(0.44);
  expect(postedConfig.scene.initial_lateral_velocity).toBe(0.12);
  expect(postedConfig.scene.initial_angular_velocity).toBe(1.7);
  expect(postedConfig.scene.launch_offset).toBe(0.015);
  expect(postedConfig.numerics.contact_time_constant).toBe(0.0045);
  expect(postedConfig.numerics.max_wall_seconds).toBe(720);
});

test("refines a three-turn preset relative to its mesh and restores it", async ({ page }) => {
  const presetConfig = {
    ...config,
    name: "3圈验证",
    material: { ...config.material, turns: 3, young_modulus: 123000000, shear_modulus: 45600000, damping: 0.000023, friction: 0.41, self_friction: 0.73 },
    numerics: { ...config.numerics, profile: "preview" as const, segments_per_turn: 16, timestep: 0.0000125 },
  };
  let postedConfig: any;
  await mockApi(page, (runConfig) => { postedConfig = runConfig; }, [{ id: "three-turn", label: "3圈验证", description: "test", config: presetConfig }]);
  await page.goto("/");
  await page.getByLabel("后端预设").selectOption("three-turn");
  await page.getByText("展开高级数值设置", { exact: true }).click();

  const segments = inputFor(page, "每圈段数");
  const timestep = inputFor(page, "积分步长");
  const physicalInputs = {
    young: inputFor(page, "杨氏模量 E"),
    shear: inputFor(page, "剪切模量 G"),
    damping: inputFor(page, "铰接阻尼"),
    friction: inputFor(page, "台阶/地面摩擦"),
    selfFriction: inputFor(page, "圈间摩擦"),
  };
  await page.locator("details.param-section").filter({ hasText: "弹性与接触" }).locator("summary").click();
  const readPhysics = async () => ({
    young: Number(await physicalInputs.young.inputValue()),
    shear: Number(await physicalInputs.shear.inputValue()),
    damping: Number(await physicalInputs.damping.inputValue()),
    friction: Number(await physicalInputs.friction.inputValue()),
    selfFriction: Number(await physicalInputs.selfFriction.inputValue()),
  });
  const initialPhysics = await readPhysics();
  await expect.poll(async () => Number(await segments.inputValue())).toBe(16);
  await expect.poll(async () => Number(await timestep.inputValue())).toBeCloseTo(0.0000125, 12);

  await page.getByRole("button", { name: "精细研究", exact: true }).click();
  await expect.poll(async () => Number(await segments.inputValue())).toBe(32);
  await expect.poll(async () => Number(await timestep.inputValue())).toBeCloseTo(0.000003125, 12);
  expect(await readPhysics()).toEqual(initialPhysics);

  await page.getByRole("button", { name: "快速预览", exact: true }).click();
  await expect.poll(async () => Number(await segments.inputValue())).toBe(16);
  await expect.poll(async () => Number(await timestep.inputValue())).toBeCloseTo(0.0000125, 12);
  expect(await readPhysics()).toEqual(initialPhysics);

  await page.getByTestId("start-run").click();
  await expect(page.getByText("已完成").first()).toBeVisible({ timeout: 10_000 });
  expect(postedConfig.numerics.profile).toBe("preview");
  expect(postedConfig.numerics.segments_per_turn).toBe(16);
  expect(postedConfig.numerics.timestep).toBeCloseTo(0.0000125, 12);
  expect(postedConfig.material.young_modulus).toBe(initialPhysics.young);
  expect(postedConfig.material.shear_modulus).toBe(initialPhysics.shear);
  expect(postedConfig.material.damping).toBe(initialPhysics.damping);
  expect(postedConfig.material.friction).toBe(initialPhysics.friction);
  expect(postedConfig.material.self_friction).toBe(initialPhysics.selfFriction);
});

test("labels ambiguous stair support without claiming confirmed flips", async ({ page }) => {
  await mockAmbiguousStairsApi(page);
  await page.goto("/");
  await page.getByTestId("start-run").click();
  await expect(page.getByTestId("result-summary")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText("翻转候选（支撑有歧义）")).toBeVisible();
  await expect(page.getByText("首次接触", { exact: true })).toBeVisible();
  await expect(page.getByText("持续端圈支撑", { exact: true })).toBeVisible();
  await expect(page.getByText("确认翻转", { exact: true })).toBeVisible();
  await expect(page.getByText("0（候选 3）", { exact: true })).toBeVisible();
  await expect(page.getByText("3 阶通过", { exact: true })).toHaveCount(0);
});

test("opens the 2D scan editor and validates scan controls", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByTestId("sweep-panel").getByRole("button", { name: /参数扫描/ }).click();
  await page.getByRole("button", { name: "2D" }).click();
  await expect(page.getByText("Y 轴")).toBeVisible();
  await expect(page.getByRole("button", { name: /开始网格扫描/ })).toBeVisible();
});

test("keeps the folded sidebars reachable on a narrow viewport", async ({ page }) => {
  await page.setViewportSize({ width: 488, height: 800 });
  await mockApi(page);
  await page.goto("/");
  await page.getByLabel("后端预设").selectOption("drop");
  await expect(page.getByTestId("scene-view")).toBeVisible();
  await expect(page.getByText("彩虹带 · 插值表面 · 几何预览（未运行物理仿真）")).toBeVisible();
  await page.waitForTimeout(1000);
  await page.getByTestId("scene-view").screenshot({ path: `${screenshotDirectory}/ui-preview-488.png` });
  await page.getByRole("button", { name: "隐藏历史" }).click();
  await expect(page.getByTestId("run-history")).toBeHidden();
  await page.getByRole("button", { name: "显示历史" }).click();
  await expect(page.getByTestId("run-history")).toBeVisible();
});
