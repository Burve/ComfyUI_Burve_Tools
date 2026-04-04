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
