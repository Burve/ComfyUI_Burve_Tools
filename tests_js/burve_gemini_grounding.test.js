const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

let helperModulePromise;

async function loadHelperModule() {
  if (!helperModulePromise) {
    const helperPath = path.join(
      __dirname,
      "..",
      "web",
      "js",
      "burve_gemini_grounding_helpers.js",
    );
    helperModulePromise = fs.readFile(helperPath, "utf8").then((source) => {
      const encoded = Buffer.from(source, "utf8").toString("base64");
      return import(`data:text/javascript;base64,${encoded}`);
    });
  }
  return helperModulePromise;
}

test("normalizes Interactions suggestions and citations", async () => {
  const { normalizeGroundingMessage } = await loadHelperModule();
  const result = normalizeGroundingMessage({
    burve_gemini_grounding: [{
      renderer_version: "burve-grounding-v1",
      metadata: {
        search_suggestions_html: ["<style>.x{}</style><div class=x>Images</div>"],
        citations: [{ title: "Source", url: "https://example.com" }],
      },
    }],
  });

  assert.equal(result.clear, false);
  assert.equal(result.suggestions.length, 1);
  assert.equal(result.citations[0].title, "Source");
});

test("preserves Google search_suggestions markup in the isolated document", async () => {
  const { buildAttributionDocument } = await loadHelperModule();
  const suggestions = "<style>.x{color:red}</style><div class=x>Exact suggestion</div>";
  const document = buildAttributionDocument({
    suggestions: [suggestions],
    citations: [],
  });

  assert.ok(document.includes(suggestions));
  assert.ok(document.includes("<base target=\"_blank\">"));
});

test("escapes citation labels and URLs", async () => {
  const { buildAttributionDocument } = await loadHelperModule();
  const document = buildAttributionDocument({
    suggestions: [],
    citations: [{ title: "<unsafe>", url: "https://example.com/?a=1&b=2" }],
  });

  assert.ok(document.includes("&lt;unsafe&gt;"));
  assert.ok(document.includes("a=1&amp;b=2"));
});

test("clears stale attribution when the backend has no grounding", async () => {
  const { normalizeGroundingMessage } = await loadHelperModule();
  assert.deepEqual(normalizeGroundingMessage({ clear: true }).suggestions, []);
  assert.equal(normalizeGroundingMessage({ clear: true }).clear, true);
});

test("flattens GenerateContent citation sources for link rendering", async () => {
  const { normalizeGroundingMessage } = await loadHelperModule();
  const result = normalizeGroundingMessage({
    burve_gemini_grounding: [{
      metadata: {
        citations: [{
          text: "supported",
          sources: [{
            title: "Primary source",
            url: "https://example.com/primary",
          }],
        }],
      },
    }],
  });

  assert.equal(result.clear, false);
  assert.equal(result.citations[0].title, "Primary source");
  assert.equal(result.citations[0].url, "https://example.com/primary");
});
