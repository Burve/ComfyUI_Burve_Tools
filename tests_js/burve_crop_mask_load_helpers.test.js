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
