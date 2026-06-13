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

test("mask inspection grayscale preserves mask opacity levels", async () => {
  const { maskAlphaToGrayscale } = await loadHelperModule();

  assert.deepEqual(
    [0, 0.25, 0.5, 0.75, 1].map(maskAlphaToGrayscale),
    [0, 64, 128, 191, 255]
  );
});

test("bucket fill stays inside a closed mask outline", async () => {
  const { buildBucketFillRuns } = await loadHelperModule();
  const width = 7;
  const height = 7;
  const mask = new Float32Array(width * height);
  for (let x = 1; x <= 5; x += 1) {
    mask[1 * width + x] = 1;
    mask[5 * width + x] = 1;
  }
  for (let y = 1; y <= 5; y += 1) {
    mask[y * width + 1] = 1;
    mask[y * width + 5] = 1;
  }
  mask[2 * width + 2] = 0.25;
  mask[3 * width + 2] = 0.75;
  mask[4 * width + 4] = 0.999;

  const result = buildBucketFillRuns({
    mask,
    imageWidth: width,
    imageHeight: height,
    crop: { x: 0, y: 0, width, height },
    seedX: 3,
    seedY: 3,
  });

  assert.equal(result.reason, null);
  assert.equal(result.pixelCount, 9);
  assert.deepEqual(result.runs, [
    [2, 2, 5],
    [3, 2, 5],
    [4, 2, 5],
  ]);
  assert.equal(mask[2 * width + 2], 1);
  assert.equal(mask[3 * width + 2], 1);
  assert.equal(mask[4 * width + 4], 1);
});

test("bucket fill leaks through an open outline", async () => {
  const { buildBucketFillRuns } = await loadHelperModule();
  const width = 7;
  const height = 7;
  const mask = new Float32Array(width * height);
  for (let x = 1; x <= 5; x += 1) {
    if (x !== 3) {
      mask[1 * width + x] = 1;
    }
    mask[5 * width + x] = 1;
  }
  for (let y = 1; y <= 5; y += 1) {
    mask[y * width + 1] = 1;
    mask[y * width + 5] = 1;
  }

  const result = buildBucketFillRuns({
    mask,
    imageWidth: width,
    imageHeight: height,
    crop: { x: 0, y: 0, width, height },
    seedX: 3,
    seedY: 3,
  });

  assert.equal(result.pixelCount, 34);
  assert.ok(result.runs.some(([y]) => y === 0));
  assert.ok(result.runs.some(([y, left, right]) => y === 3 && left === 2 && right === 5));
});

test("bucket fill is clipped to the crop boundary", async () => {
  const { buildBucketFillRuns } = await loadHelperModule();
  const result = buildBucketFillRuns({
    mask: new Float32Array(25),
    imageWidth: 5,
    imageHeight: 5,
    crop: { x: 1, y: 1, width: 3, height: 3 },
    seedX: 2,
    seedY: 2,
  });

  assert.equal(result.pixelCount, 9);
  assert.deepEqual(result.runs, [
    [1, 1, 4],
    [2, 1, 4],
    [3, 1, 4],
  ]);
});

test("four-connected bucket fill does not cross a diagonal wall", async () => {
  const { buildBucketFillRuns } = await loadHelperModule();
  const width = 5;
  const height = 5;
  const mask = new Float32Array(width * height);
  for (let index = 0; index < 5; index += 1) {
    mask[index * width + index] = 1;
  }

  const result = buildBucketFillRuns({
    mask,
    imageWidth: width,
    imageHeight: height,
    crop: { x: 0, y: 0, width, height },
    seedX: 4,
    seedY: 0,
  });

  assert.equal(result.pixelCount, 10);
  assert.ok(result.runs.every(([y, left]) => left > y));
});

test("bucket fill traverses a partially masked seed and upgrades feather pixels", async () => {
  const { buildBucketFillRuns } = await loadHelperModule();
  const mask = new Float32Array([1, 0.25, 0.5, 0.999, 1]);

  const result = buildBucketFillRuns({
    mask,
    imageWidth: 5,
    imageHeight: 1,
    crop: { x: 0, y: 0, width: 5, height: 1 },
    seedX: 2,
    seedY: 0,
  });

  assert.deepEqual(result, { runs: [[0, 1, 4]], pixelCount: 3, reason: null });
  assert.deepEqual(Array.from(mask), [1, 1, 1, 1, 1]);
});

test("bucket fill treats effectively full alpha as a solid boundary", async () => {
  const { buildBucketFillRuns, BUCKET_BOUNDARY_ALPHA } = await loadHelperModule();
  const mask = new Float32Array(3);
  mask[1] = 0.999999;

  const result = buildBucketFillRuns({
    mask,
    imageWidth: 3,
    imageHeight: 1,
    crop: { x: 0, y: 0, width: 3, height: 1 },
    seedX: 1,
    seedY: 0,
  });

  assert.equal(mask[1], BUCKET_BOUNDARY_ALPHA);
  assert.deepEqual(result, { runs: [], pixelCount: 0, reason: "masked" });
});

test("source mask replay clips bucket runs and applies later erase operations", async () => {
  const { renderSourceMask } = await loadHelperModule();
  const mask = renderSourceMask({
    imageWidth: 6,
    imageHeight: 5,
    crop: { x: 1, y: 1, width: 4, height: 3 },
    operations: [
      {
        mode: "bucket",
        points: [[2, 2]],
        seed: [2, 2],
        runs: [
          [0, 0, 6],
          [1, 0, 6],
          [2, 0, 6],
          [3, 0, 6],
          [4, 0, 6],
        ],
      },
      {
        mode: "erase",
        radius_px: 1,
        softness: 0,
        opacity: 1,
        points: [[3, 2]],
      },
    ],
  });

  assert.equal(mask[0 * 6 + 2], 0);
  assert.equal(mask[1 * 6 + 1], 1);
  assert.equal(mask[2 * 6 + 1], 1);
  assert.equal(mask[2 * 6 + 2], 0);
  assert.equal(mask[2 * 6 + 3], 0);
  assert.equal(mask[2 * 6 + 4], 0);
  assert.equal(mask[3 * 6 + 4], 1);
  assert.equal(mask[4 * 6 + 2], 0);
});
