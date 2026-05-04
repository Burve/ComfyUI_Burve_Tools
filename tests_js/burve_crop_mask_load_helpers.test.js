const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

let helperModulePromise;

async function loadHelperModule() {
  if (!helperModulePromise) {
    const helperPath = path.join(__dirname, "..", "web", "js", "burve_crop_mask_load_helpers.js");
    helperModulePromise = fs.readFile(helperPath, "utf8").then((source) => {
      const encoded = Buffer.from(source, "utf8").toString("base64");
      return import(`data:text/javascript;base64,${encoded}`);
    });
  }
  return helperModulePromise;
}

test("buildImageWidgetValue keeps root files at the top level", async () => {
  const { buildImageWidgetValue } = await loadHelperModule();
  assert.equal(buildImageWidgetValue({ name: "x.png", subfolder: "" }), "x.png");
});

test("buildImageWidgetValue prefixes nested subfolders", async () => {
  const { buildImageWidgetValue } = await loadHelperModule();
  assert.equal(buildImageWidgetValue({ name: "x.png", subfolder: "nested" }), "nested/x.png");
});

test("computeCropMaskWidgetLayout keeps default widget height stable across repeated layouts", async () => {
  const { computeCropMaskWidgetLayout } = await loadHelperModule();
  const first = computeCropMaskWidgetLayout({
    widgetHeight: 768,
    chromeHeight: 318,
    minCanvasHeight: 420,
  });
  const second = computeCropMaskWidgetLayout({
    widgetHeight: first.widgetHeight,
    chromeHeight: 318,
    minCanvasHeight: 420,
  });

  assert.deepEqual(first, { widgetHeight: 768, canvasHeight: 450 });
  assert.deepEqual(second, first);
});

test("computeCropMaskWidgetLayout grows only to fit larger chrome", async () => {
  const { computeCropMaskWidgetLayout } = await loadHelperModule();
  const first = computeCropMaskWidgetLayout({
    widgetHeight: 768,
    chromeHeight: 390,
    minCanvasHeight: 420,
  });
  const second = computeCropMaskWidgetLayout({
    widgetHeight: first.widgetHeight,
    chromeHeight: 390,
    minCanvasHeight: 420,
  });

  assert.deepEqual(first, { widgetHeight: 810, canvasHeight: 420 });
  assert.deepEqual(second, first);
});

test("computeCropMaskWidgetLayout ignores parent DOM height inputs", async () => {
  const { computeCropMaskWidgetLayout } = await loadHelperModule();
  const layout = computeCropMaskWidgetLayout({
    widgetHeight: 768,
    chromeHeight: 318,
    minCanvasHeight: 420,
    parentHeight: 10000,
  });

  assert.deepEqual(layout, { widgetHeight: 768, canvasHeight: 450 });
});

test("computeCropMaskWidgetLayout keeps canvas height at least one pixel", async () => {
  const { computeCropMaskWidgetLayout } = await loadHelperModule();
  const layout = computeCropMaskWidgetLayout({
    widgetHeight: 10,
    chromeHeight: 50,
    minCanvasHeight: -100,
  });

  assert.deepEqual(layout, { widgetHeight: 10, canvasHeight: 1 });
});

test("isClientPointInsideRect returns true for center and boundary points", async () => {
  const { isClientPointInsideRect } = await loadHelperModule();
  const rect = { left: 10, top: 20, width: 100, height: 80 };

  assert.equal(isClientPointInsideRect({ clientX: 60, clientY: 60, rect }), true);
  assert.equal(isClientPointInsideRect({ clientX: 10, clientY: 20, rect }), true);
  assert.equal(isClientPointInsideRect({ clientX: 110, clientY: 100, rect }), true);
});

test("isClientPointInsideRect returns false outside the rect", async () => {
  const { isClientPointInsideRect } = await loadHelperModule();
  const rect = { left: 10, top: 20, width: 100, height: 80 };

  assert.equal(isClientPointInsideRect({ clientX: 9.99, clientY: 60, rect }), false);
  assert.equal(isClientPointInsideRect({ clientX: 110.01, clientY: 60, rect }), false);
  assert.equal(isClientPointInsideRect({ clientX: 60, clientY: 19.99, rect }), false);
  assert.equal(isClientPointInsideRect({ clientX: 60, clientY: 100.01, rect }), false);
});

test("isClientPointInsideRect returns false for missing or empty rects", async () => {
  const { isClientPointInsideRect } = await loadHelperModule();

  assert.equal(isClientPointInsideRect({ clientX: 60, clientY: 60, rect: null }), false);
  assert.equal(isClientPointInsideRect({ clientX: 60, clientY: 60, rect: { left: 10, top: 20, width: 0, height: 80 } }), false);
  assert.equal(isClientPointInsideRect({ clientX: 60, clientY: 60, rect: { left: 10, top: 20, width: 100, height: 0 } }), false);
});

test("filterImageOptions returns all options for an empty query", async () => {
  const { filterImageOptions } = await loadHelperModule();
  assert.deepEqual(filterImageOptions({ nextOptions: ["a.png", "nested/b.png"], query: "" }), ["a.png", "nested/b.png"]);
});

test("filterImageOptions matches root filenames", async () => {
  const { filterImageOptions } = await loadHelperModule();
  assert.deepEqual(
    filterImageOptions({
      nextOptions: ["portrait.png", "nested/landscape.png", "mask.webp"],
      query: "mask",
    }),
    ["mask.webp"]
  );
});

test("filterImageOptions matches nested paths", async () => {
  const { filterImageOptions } = await loadHelperModule();
  assert.deepEqual(
    filterImageOptions({
      nextOptions: ["portrait.png", "nested/landscape.png", "nested/deeper/mask.webp"],
      query: "nested/deeper",
    }),
    ["nested/deeper/mask.webp"]
  );
});

test("filterImageOptions is case-insensitive", async () => {
  const { filterImageOptions } = await loadHelperModule();
  assert.deepEqual(
    filterImageOptions({
      nextOptions: ["Portrait.PNG", "nested/Landscape.png", "mask.webp"],
      query: "portrait",
    }),
    ["Portrait.PNG"]
  );
});

test("selectSyncedImageValue keeps the current selection when it still exists", async () => {
  const { selectSyncedImageValue } = await loadHelperModule();
  assert.equal(
    selectSyncedImageValue({
      currentValue: "keep.png",
      nextOptions: ["new.png", "keep.png"],
    }),
    "keep.png"
  );
});

test("selectSyncedImageValue prefers the uploaded image when present", async () => {
  const { selectSyncedImageValue } = await loadHelperModule();
  assert.equal(
    selectSyncedImageValue({
      preferredValue: "uploaded.png",
      currentValue: "keep.png",
      nextOptions: ["uploaded.png", "keep.png"],
    }),
    "uploaded.png"
  );
});

test("selectSyncedImageValue falls back to the first option when the current selection disappeared", async () => {
  const { selectSyncedImageValue } = await loadHelperModule();
  assert.equal(
    selectSyncedImageValue({
      currentValue: "missing.png",
      nextOptions: ["first.png", "second.png"],
    }),
    "first.png"
  );
});

test("selectSyncedImageValue retains the current selection after filtering and sync", async () => {
  const { filterImageOptions, selectSyncedImageValue } = await loadHelperModule();
  const filteredOptions = filterImageOptions({
    nextOptions: ["portrait.png", "nested/keep.png", "other.png"],
    query: "keep",
  });
  assert.equal(
    selectSyncedImageValue({
      currentValue: "nested/keep.png",
      nextOptions: filteredOptions,
    }),
    "nested/keep.png"
  );
});
