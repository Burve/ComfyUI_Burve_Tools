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

test("computeCropMaskWidgetLayout gives extra editor height to the canvas", async () => {
  const { computeCropMaskWidgetLayout } = await loadHelperModule();
  const first = computeCropMaskWidgetLayout({
    availableHeight: 900,
    chromeHeight: 318,
    minCanvasHeight: 420,
  });
  const second = computeCropMaskWidgetLayout({
    availableHeight: first.widgetHeight,
    chromeHeight: 318,
    minCanvasHeight: 420,
  });

  assert.deepEqual(first, { minWidgetHeight: 738, widgetHeight: 900, canvasHeight: 582 });
  assert.deepEqual(second, first);
});

test("computeCropMaskWidgetLayout shrinks the canvas when available height decreases", async () => {
  const { computeCropMaskWidgetLayout } = await loadHelperModule();
  const tall = computeCropMaskWidgetLayout({
    availableHeight: 1100,
    chromeHeight: 318,
    minCanvasHeight: 420,
  });
  const shorter = computeCropMaskWidgetLayout({
    availableHeight: 800,
    chromeHeight: 318,
    minCanvasHeight: 420,
  });

  assert.equal(tall.canvasHeight, 782);
  assert.deepEqual(shorter, { minWidgetHeight: 738, widgetHeight: 800, canvasHeight: 482 });
});

test("computeCropMaskWidgetLayout raises the minimum for wrapped chrome", async () => {
  const { computeCropMaskWidgetLayout } = await loadHelperModule();
  const layout = computeCropMaskWidgetLayout({
    availableHeight: 768,
    chromeHeight: 390,
    minCanvasHeight: 420,
  });

  assert.deepEqual(layout, { minWidgetHeight: 810, widgetHeight: 810, canvasHeight: 420 });
});

test("computeCropMaskWidgetLayout keeps canvas height at least one pixel", async () => {
  const { computeCropMaskWidgetLayout } = await loadHelperModule();
  const layout = computeCropMaskWidgetLayout({
    availableHeight: 10,
    chromeHeight: 50,
    minCanvasHeight: -100,
  });

  assert.deepEqual(layout, { minWidgetHeight: 51, widgetHeight: 51, canvasHeight: 1 });
});

test("hideCropMaskNativeWidgets hides only auxiliary image widgets", async () => {
  const { hideCropMaskNativeWidgets } = await loadHelperModule();
  const callback = () => {};
  const imageWidget = { name: "image", value: "keep.png", callback, serialize: true };
  const uploadWidget = { name: "upload" };
  const refreshWidget = { name: "refresh" };
  const autoRefreshWidget = { name: "Auto-refresh after generation", value: false };
  const ratioWidget = { name: "aspect_ratio", value: "9:16" };
  const node = { widgets: [imageWidget, uploadWidget, refreshWidget, autoRefreshWidget, ratioWidget] };

  assert.equal(hideCropMaskNativeWidgets(node), 4);
  for (const widget of [imageWidget, uploadWidget, refreshWidget, autoRefreshWidget]) {
    assert.equal(widget.hidden, true);
    assert.deepEqual(widget.computeSize(), [0, -4]);
  }
  assert.equal(imageWidget.value, "keep.png");
  assert.equal(imageWidget.callback, callback);
  assert.equal(imageWidget.serialize, true);
  assert.equal(ratioWidget.hidden, undefined);
  assert.equal(ratioWidget.computeSize, undefined);
});

test("native preview guard replaces an existing preview through node APIs", async () => {
  const { ensureNativeImagePreviewGuard, isNativeImagePreviewGuard } = await loadHelperModule();
  let cleanupCount = 0;
  const keepWidget = { name: "burve_crop_mask_editor" };
  const previewWidget = {
    name: "$$canvas-image-preview",
    onRemove() {
      cleanupCount += 1;
    },
  };
  const node = {
    widgets: [keepWidget, previewWidget],
    imgs: [{ src: "preview.png" }],
    imageIndex: 2,
    overIndex: 1,
    removeWidget(widget) {
      const index = this.widgets.indexOf(widget);
      assert.notEqual(index, -1);
      widget.onRemove?.();
      this.widgets.splice(index, 1);
      this._widgetSlotsDirty = true;
    },
    addCustomWidget(widget) {
      this.widgets.push(widget);
      this._widgetSlotsDirty = true;
      return widget;
    },
  };

  const first = ensureNativeImagePreviewGuard(node);
  assert.equal(first.added, true);
  assert.equal(first.removedCount, 1);
  assert.equal(cleanupCount, 1);
  assert.equal(node.widgets[0], keepWidget);
  assert.equal(isNativeImagePreviewGuard(first.guard), true);
  assert.equal(first.guard.name, "$$canvas-image-preview");
  assert.equal(first.guard.hidden, true);
  assert.equal(first.guard.serialize, false);
  assert.deepEqual(first.guard.computeSize(), [0, -4]);
  assert.deepEqual(first.guard.computeLayoutSize(), {
    minHeight: 0,
    maxHeight: 0,
    minWidth: 0,
    maxWidth: 0,
  });
  assert.deepEqual(node.imgs, [{ src: "preview.png" }]);
  assert.equal(node.imageIndex, 2);
  assert.equal(node.overIndex, 1);

  node.imgs = [{ src: "later-preview.png" }];
  const second = ensureNativeImagePreviewGuard(node);
  assert.equal(second.guard, first.guard);
  assert.equal(second.added, false);
  assert.equal(second.removedCount, 0);
  assert.equal(node.widgets.filter((widget) => widget.name === "$$canvas-image-preview").length, 1);
  assert.equal(cleanupCount, 1);
});

test("native preview guard can be removed without disturbing unrelated widgets", async () => {
  const { ensureNativeImagePreviewGuard, removeNativeImagePreviewGuard } = await loadHelperModule();
  const keepWidget = { name: "aspect_ratio" };
  const node = {
    widgets: [keepWidget],
    removeWidget(widget) {
      this.widgets.splice(this.widgets.indexOf(widget), 1);
    },
    addCustomWidget(widget) {
      this.widgets.push(widget);
      return widget;
    },
  };
  const { guard } = ensureNativeImagePreviewGuard(node);

  assert.equal(removeNativeImagePreviewGuard(node, guard), true);
  assert.deepEqual(node.widgets, [keepWidget]);
  assert.equal(removeNativeImagePreviewGuard(node, guard), false);
});

test("DOM widget geometry accepts matching logical size and graph scale", async () => {
  const { evaluateDomWidgetGeometry } = await loadHelperModule();
  const geometry = evaluateDomWidgetGeometry({
    nodeWidth: 580,
    widgetHeight: 760,
    margin: 10,
    hostClientWidth: 560,
    hostClientHeight: 740,
    hostRectWidth: 280,
    hostRectHeight: 370,
    graphScale: 0.5,
  });

  assert.equal(geometry.status, "healthy");
  assert.equal(geometry.widthMatches, true);
  assert.equal(geometry.heightMatches, true);
  assert.equal(geometry.scaleMatches, true);
});

test("DOM widget geometry detects half-width allocation and stale transform scale", async () => {
  const { evaluateDomWidgetGeometry } = await loadHelperModule();
  const halfWidth = evaluateDomWidgetGeometry({
    nodeWidth: 580,
    widgetHeight: 760,
    margin: 10,
    hostClientWidth: 280,
    hostClientHeight: 740,
    hostRectWidth: 140,
    hostRectHeight: 370,
    graphScale: 0.5,
  });
  const staleTransform = evaluateDomWidgetGeometry({
    nodeWidth: 580,
    widgetHeight: 760,
    margin: 10,
    hostClientWidth: 560,
    hostClientHeight: 740,
    hostRectWidth: 140,
    hostRectHeight: 185,
    graphScale: 0.5,
  });

  assert.equal(halfWidth.status, "mismatch");
  assert.equal(halfWidth.widthMatches, false);
  assert.equal(halfWidth.scaleMatches, true);
  assert.equal(staleTransform.status, "mismatch");
  assert.equal(staleTransform.widthMatches, true);
  assert.equal(staleTransform.scaleMatches, false);
});

test("DOM widget geometry ignores unmounted and zero-sized transient hosts", async () => {
  const { evaluateDomWidgetGeometry } = await loadHelperModule();
  const geometry = evaluateDomWidgetGeometry({
    nodeWidth: 580,
    widgetHeight: 760,
    margin: 10,
    hostClientWidth: 0,
    hostClientHeight: 0,
    hostRectWidth: 0,
    hostRectHeight: 0,
    graphScale: 0.5,
  });

  assert.equal(geometry.status, "transient");
});

test("DOM widget recovery reflows once, remounts once, and resets when healthy", async () => {
  const { nextDomWidgetRecoveryState } = await loadHelperModule();
  const first = nextDomWidgetRecoveryState({ geometryStatus: "mismatch" });
  const second = nextDomWidgetRecoveryState({
    geometryStatus: "mismatch",
    mismatchFrames: first.mismatchFrames,
    remounted: first.remounted,
  });
  const third = nextDomWidgetRecoveryState({
    geometryStatus: "mismatch",
    mismatchFrames: second.mismatchFrames,
    remounted: second.remounted,
  });
  const healthy = nextDomWidgetRecoveryState({
    geometryStatus: "healthy",
    mismatchFrames: third.mismatchFrames,
    remounted: third.remounted,
  });

  assert.deepEqual(first, { action: "reflow", mismatchFrames: 1, remounted: false });
  assert.deepEqual(second, { action: "remount", mismatchFrames: 2, remounted: true });
  assert.deepEqual(third, { action: "none", mismatchFrames: 3, remounted: true });
  assert.deepEqual(healthy, { action: "none", mismatchFrames: 0, remounted: false });
});

test("DOM widget replacement preserves node dimensions and external editor state", async () => {
  const { replaceDomWidgetPreservingNode } = await loadHelperModule();
  const currentWidget = { name: "burve_crop_mask_editor" };
  const nextWidget = { name: "burve_crop_mask_editor" };
  const editorState = {
    crop: { x: 10, y: 20, width: 300, height: 400 },
    viewport: { zoom: 1.75, pan_x: 8, pan_y: -3 },
    strokes: [{ mode: "paint", points: [[1, 2]] }],
  };
  const serializedBefore = JSON.stringify(editorState);
  let restoredHooks = false;
  const node = {
    size: new Float32Array([644, 1122]),
    widgets: [currentWidget],
    removeWidget(widget) {
      this.widgets.splice(this.widgets.indexOf(widget), 1);
      this.size = [320, 500];
    },
    setSize(size) {
      this.size = [...size];
    },
  };

  const result = replaceDomWidgetPreservingNode({
    node,
    currentWidget,
    beforeRemove() {
      restoredHooks = true;
    },
    createWidget() {
      node.widgets.push(nextWidget);
      return nextWidget;
    },
  });

  assert.equal(result, nextWidget);
  assert.equal(restoredHooks, true);
  assert.deepEqual(node.size, [644, 1122]);
  assert.deepEqual(node.widgets, [nextWidget]);
  assert.equal(node._widgetSlotsDirty, true);
  assert.equal(JSON.stringify(editorState), serializedBefore);
});

test("crop-mask lifecycle never mutates preview widgets from background drawing", async () => {
  const frontendPath = path.join(__dirname, "..", "web", "js", "burve_crop_mask_load.js");
  const helperPath = path.join(__dirname, "..", "web", "js", "burve_crop_mask_load_helpers.js");
  const [frontendSource, helperSource] = await Promise.all([
    fs.readFile(frontendPath, "utf8"),
    fs.readFile(helperPath, "utf8"),
  ]);

  assert.doesNotMatch(frontendSource, /wrap\("onDrawBackground"/);
  assert.doesNotMatch(frontendSource, /scheduleNativeImageUiSuppression/);
  assert.doesNotMatch(helperSource, /node\.widgets\.splice/);
  assert.doesNotMatch(helperSource, /node\.imgs\s*=/);
});

test("crop-mask workflow values migrate before LiteGraph configure restores widgets", async () => {
  const frontendPath = path.join(__dirname, "..", "web", "js", "burve_crop_mask_load.js");
  const frontendSource = await fs.readFile(frontendPath, "utf8");

  assert.match(
    frontendSource,
    /configure\?\.call\(this, migrateCropMaskSerializedNode\(serializedNode\)\)/
  );
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

test("crop-mask workflow migration removes serialized remote toggle and refresh slots", async () => {
  const { migrateCropMaskSerializedNode } = await loadHelperModule();
  const editorState = JSON.stringify({ image_name: "portrait.webp [input]", crop: { x: 1, y: 2 } });
  const serializedNode = {
    id: 46,
    widgets_values: ["portrait.webp [input]", false, "refresh", "9:16", editorState, "image"],
  };

  const migrated = migrateCropMaskSerializedNode(serializedNode);

  assert.notEqual(migrated, serializedNode);
  assert.deepEqual(migrated.widgets_values, ["portrait.webp [input]", "9:16", editorState]);
  assert.deepEqual(serializedNode.widgets_values, [
    "portrait.webp [input]",
    false,
    "refresh",
    "9:16",
    editorState,
    "image",
  ]);
});

test("crop-mask workflow migration accepts an enabled legacy remote toggle", async () => {
  const { migrateCropMaskSerializedNode } = await loadHelperModule();
  const serializedNode = {
    widgets_values: ["portrait.webp [input]", true, "refresh", "1:1", "", "image"],
  };

  assert.deepEqual(migrateCropMaskSerializedNode(serializedNode).widgets_values, [
    "portrait.webp [input]",
    "1:1",
    "",
  ]);
});

test("crop-mask workflow migration removes compact remote refresh slots", async () => {
  const { migrateCropMaskSerializedNode } = await loadHelperModule();
  const editorState = JSON.stringify({ image_name: "nested/image.png [input]" });
  const serializedNode = {
    widgets_values: ["nested/image.png [input]", "refresh", "4:3", editorState],
  };

  assert.deepEqual(migrateCropMaskSerializedNode(serializedNode).widgets_values, [
    "nested/image.png [input]",
    "4:3",
    editorState,
  ]);
});

test("crop-mask workflow migration leaves current and malformed layouts unchanged", async () => {
  const { migrateCropMaskSerializedNode } = await loadHelperModule();
  const current = { widgets_values: ["image.webp [input]", "1:1", "{}"] };
  const malformed = { widgets_values: ["image.webp [input]", false, "refresh", 1, {}] };

  assert.equal(migrateCropMaskSerializedNode(current), current);
  assert.equal(migrateCropMaskSerializedNode(malformed), malformed);
});

test("crop-mask workflow migration defers to named widget values", async () => {
  const { migrateCropMaskSerializedNode } = await loadHelperModule();
  const serializedNode = {
    widgets_values: ["legacy.webp [input]", false, "refresh", "1:1", "{}", "image"],
    widgets_values_named: {
      image: "named.webp [input]",
      aspect_ratio: "16:9",
      editor_state_json: "{}",
    },
  };

  assert.equal(migrateCropMaskSerializedNode(serializedNode), serializedNode);
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
