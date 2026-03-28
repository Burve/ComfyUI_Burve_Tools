const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

let geometryModulePromise;

async function loadGeometryModule() {
  if (!geometryModulePromise) {
    const helperPath = path.join(__dirname, "..", "web", "js", "burve_crop_mask_geometry.js");
    geometryModulePromise = fs.readFile(helperPath, "utf8").then((source) => {
      const encoded = Buffer.from(source, "utf8").toString("base64");
      return import(`data:text/javascript;base64,${encoded}`);
    });
  }
  return geometryModulePromise;
}

function assertPointClose(actual, expected, epsilon = 1e-9) {
  assert.ok(Math.abs(actual.x - expected.x) <= epsilon, `expected x=${expected.x}, got ${actual.x}`);
  assert.ok(Math.abs(actual.y - expected.y) <= epsilon, `expected y=${expected.y}, got ${actual.y}`);
}

function projectImagePoint({
  stageWidth,
  stageHeight,
  imageWidth,
  imageHeight,
  fitScale,
  viewport,
  imagePoint,
}) {
  const scale = fitScale * viewport.zoom;
  return {
    x: (stageWidth - imageWidth * scale) / 2 + viewport.pan_x * scale + imagePoint.x * scale,
    y: (stageHeight - imageHeight * scale) / 2 + viewport.pan_y * scale + imagePoint.y * scale,
  };
}

test("getLogicalCanvasMetrics reports display-to-logical scale factors", async () => {
  const { getLogicalCanvasMetrics } = await loadGeometryModule();
  const metrics = getLogicalCanvasMetrics({
    canvas: {
      clientWidth: 400,
      clientHeight: 240,
      getBoundingClientRect() {
        return { left: 10, top: 20, width: 200, height: 120 };
      },
    },
    logicalWidth: 400,
    logicalHeight: 240,
  });

  assert.equal(metrics.rect.left, 10);
  assert.equal(metrics.rect.top, 20);
  assert.equal(metrics.scaleX, 2);
  assert.equal(metrics.scaleY, 2);
});

test("clientToLogicalCanvasPoint returns identity at 1x display scale", async () => {
  const { clientToLogicalCanvasPoint } = await loadGeometryModule();
  const point = clientToLogicalCanvasPoint({
    clientX: 170,
    clientY: 120,
    rect: { left: 100, top: 50, width: 400, height: 300 },
    logicalWidth: 400,
    logicalHeight: 300,
  });

  assert.deepEqual(point, { x: 70, y: 70 });
});

test("clientToLogicalCanvasPoint maps correctly at 0.5x and 2x display scale", async () => {
  const { clientToLogicalCanvasPoint } = await loadGeometryModule();

  const zoomedOut = clientToLogicalCanvasPoint({
    clientX: 300,
    clientY: 150,
    rect: { left: 100, top: 50, width: 200, height: 100 },
    logicalWidth: 400,
    logicalHeight: 200,
  });
  const zoomedIn = clientToLogicalCanvasPoint({
    clientX: 300,
    clientY: 150,
    rect: { left: 100, top: 50, width: 800, height: 400 },
    logicalWidth: 400,
    logicalHeight: 200,
  });

  assert.deepEqual(zoomedOut, { x: 400, y: 200 });
  assert.deepEqual(zoomedIn, { x: 100, y: 50 });
});

test("clientToLogicalCanvasPoint handles fractional rect sizes without drift", async () => {
  const { clientToLogicalCanvasPoint, logicalToClientCanvasPoint } = await loadGeometryModule();
  const rect = { left: 12.5, top: 18.25, width: 333.3, height: 222.2 };
  const logicalPoint = clientToLogicalCanvasPoint({
    clientX: 179.15,
    clientY: 129.35,
    rect,
    logicalWidth: 1000,
    logicalHeight: 500,
  });
  const roundTripped = logicalToClientCanvasPoint({
    x: logicalPoint.x,
    y: logicalPoint.y,
    rect,
    logicalWidth: 1000,
    logicalHeight: 500,
  });

  assertPointClose(roundTripped, { x: 179.15, y: 129.35 }, 1e-6);
});

test("computeZoomAnchoredViewport preserves the anchored image point when zooming in and out", async () => {
  const { computeZoomAnchoredViewport } = await loadGeometryModule();
  const baseConfig = {
    stageWidth: 620,
    stageHeight: 410,
    imageWidth: 1024,
    imageHeight: 768,
    fitScale: 0.42,
  };
  const initialViewport = { zoom: 1, pan_x: 36, pan_y: -18 };
  const imagePoint = { x: 420, y: 250 };
  const anchorCanvasPoint = projectImagePoint({
    ...baseConfig,
    viewport: initialViewport,
    imagePoint,
  });

  for (const nextZoom of [1.75, 0.65]) {
    const nextViewport = computeZoomAnchoredViewport({
      ...baseConfig,
      nextZoom,
      anchorCanvasPoint,
      anchorImagePoint: imagePoint,
    });
    const projected = projectImagePoint({
      ...baseConfig,
      viewport: nextViewport,
      imagePoint,
    });

    assert.equal(nextViewport.zoom, nextZoom);
    assertPointClose(projected, anchorCanvasPoint, 1e-6);
  }
});
