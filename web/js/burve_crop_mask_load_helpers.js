export function buildImageWidgetValue(uploadResult) {
  const name = String(uploadResult?.name || "").trim();
  const subfolder = String(uploadResult?.subfolder || "")
    .replace(/\\/g, "/")
    .replace(/^\/+|\/+$/g, "")
    .trim();
  if (!name) {
    return "";
  }
  return subfolder ? `${subfolder}/${name}` : name;
}

export function computeCropMaskWidgetLayout({ availableHeight, chromeHeight, minCanvasHeight }) {
  const resolvedChromeHeight = Math.max(0, Math.ceil(Number(chromeHeight) || 0));
  const resolvedMinCanvasHeight = Math.max(1, Math.ceil(Number(minCanvasHeight) || 0));
  const minWidgetHeight = resolvedChromeHeight + resolvedMinCanvasHeight;
  const requestedHeight = Math.floor(Number(availableHeight));
  const widgetHeight = Math.max(
    minWidgetHeight,
    Number.isFinite(requestedHeight) ? requestedHeight : minWidgetHeight
  );
  return {
    minWidgetHeight,
    widgetHeight,
    canvasHeight: Math.max(resolvedMinCanvasHeight, widgetHeight - resolvedChromeHeight),
  };
}

const CROP_MASK_NATIVE_WIDGET_NAMES = new Set([
  "image",
  "upload",
  "refresh",
  "Auto-refresh after generation",
]);

export const NATIVE_IMAGE_PREVIEW_WIDGET_NAME = "$$canvas-image-preview";
const NATIVE_IMAGE_PREVIEW_GUARD_MARKER = "__burveNativeImagePreviewGuard";
const HIDDEN_WIDGET_SIZE = () => [0, -4];

export function createNativeImagePreviewGuard() {
  return {
    name: NATIVE_IMAGE_PREVIEW_WIDGET_NAME,
    type: "BURVE_NATIVE_IMAGE_PREVIEW_GUARD",
    value: "",
    hidden: true,
    serialize: false,
    options: { serialize: false, canvasOnly: true },
    [NATIVE_IMAGE_PREVIEW_GUARD_MARKER]: true,
    computeSize: HIDDEN_WIDGET_SIZE,
    computeLayoutSize: () => ({ minHeight: 0, maxHeight: 0, minWidth: 0, maxWidth: 0 }),
    draw: () => {},
    drawWidget: () => {},
    onClick: () => {},
  };
}

export function isNativeImagePreviewGuard(widget) {
  return Boolean(widget?.[NATIVE_IMAGE_PREVIEW_GUARD_MARKER]);
}

export function ensureNativeImagePreviewGuard(node) {
  if (!node || !Array.isArray(node.widgets)) {
    return { guard: null, added: false, removedCount: 0 };
  }

  let guard = node.widgets.find(isNativeImagePreviewGuard) ?? null;
  let removedCount = 0;
  const stalePreviews = node.widgets.filter(
    (widget) => widget?.name === NATIVE_IMAGE_PREVIEW_WIDGET_NAME && widget !== guard
  );
  for (const previewWidget of stalePreviews) {
    if (typeof node.removeWidget !== "function") {
      continue;
    }
    node.removeWidget(previewWidget);
    removedCount += 1;
  }

  if (guard || typeof node.addCustomWidget !== "function") {
    return { guard, added: false, removedCount };
  }

  guard = node.addCustomWidget(createNativeImagePreviewGuard());
  return { guard, added: true, removedCount };
}

export function removeNativeImagePreviewGuard(node, guard) {
  if (!node || !guard || !isNativeImagePreviewGuard(guard) || !node.widgets?.includes?.(guard)) {
    return false;
  }
  if (typeof node.removeWidget !== "function") {
    return false;
  }
  node.removeWidget(guard);
  return true;
}

export function hideCropMaskNativeWidgets(node) {
  let hiddenCount = 0;
  for (const widget of node?.widgets ?? []) {
    if (!CROP_MASK_NATIVE_WIDGET_NAMES.has(widget?.name)) {
      continue;
    }
    widget.hidden = true;
    widget.computeSize = HIDDEN_WIDGET_SIZE;
    hiddenCount += 1;
  }
  return hiddenCount;
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function evaluateDomWidgetGeometry({
  nodeWidth,
  widgetWidth,
  widgetHeight,
  margin,
  hostClientWidth,
  hostClientHeight,
  hostRectWidth,
  hostRectHeight,
  graphScale,
  logicalTolerance = 2,
  scaleTolerance = 0.025,
}) {
  const resolvedNodeWidth = finiteNumber(nodeWidth);
  const resolvedWidgetWidth = finiteNumber(widgetWidth);
  const resolvedWidgetHeight = finiteNumber(widgetHeight);
  const resolvedMargin = Math.max(0, finiteNumber(margin) ?? 0);
  const logicalWidth = finiteNumber(hostClientWidth);
  const logicalHeight = finiteNumber(hostClientHeight);
  const renderedWidth = finiteNumber(hostRectWidth);
  const renderedHeight = finiteNumber(hostRectHeight);
  const expectedScale = finiteNumber(graphScale);
  const expectedWidth = (resolvedWidgetWidth ?? resolvedNodeWidth ?? 0) - resolvedMargin * 2;
  const expectedHeight = (resolvedWidgetHeight ?? 0) - resolvedMargin * 2;

  if (
    !(expectedWidth > 0) ||
    !(expectedHeight > 0) ||
    !(logicalWidth > 0) ||
    !(logicalHeight > 0) ||
    !(renderedWidth > 0) ||
    !(renderedHeight > 0)
  ) {
    return { status: "transient", healthy: false };
  }

  const actualScaleX = renderedWidth / logicalWidth;
  const actualScaleY = renderedHeight / logicalHeight;
  const widthMatches = Math.abs(logicalWidth - expectedWidth) <= logicalTolerance;
  const heightMatches = Math.abs(logicalHeight - expectedHeight) <= logicalTolerance;
  const scaleMatches =
    !(expectedScale > 0) ||
    (Math.abs(actualScaleX - expectedScale) <= scaleTolerance &&
      Math.abs(actualScaleY - expectedScale) <= scaleTolerance);
  const healthy = widthMatches && heightMatches && scaleMatches;

  return {
    status: healthy ? "healthy" : "mismatch",
    healthy,
    expectedWidth,
    expectedHeight,
    logicalWidth,
    logicalHeight,
    actualScaleX,
    actualScaleY,
    expectedScale,
    widthMatches,
    heightMatches,
    scaleMatches,
  };
}

export function nextDomWidgetRecoveryState({ geometryStatus, mismatchFrames = 0, remounted = false }) {
  if (geometryStatus === "healthy") {
    return { action: "none", mismatchFrames: 0, remounted: false };
  }
  if (geometryStatus !== "mismatch") {
    return { action: "none", mismatchFrames, remounted };
  }

  const nextMismatchFrames = mismatchFrames + 1;
  if (nextMismatchFrames === 1) {
    return { action: "reflow", mismatchFrames: nextMismatchFrames, remounted };
  }
  if (!remounted) {
    return { action: "remount", mismatchFrames: nextMismatchFrames, remounted: true };
  }
  return { action: "none", mismatchFrames: nextMismatchFrames, remounted };
}

export function replaceDomWidgetPreservingNode({ node, currentWidget, beforeRemove, createWidget }) {
  if (!node || !currentWidget || typeof node.removeWidget !== "function" || typeof createWidget !== "function") {
    throw new TypeError("DOM widget replacement requires a node, current widget, and supported widget APIs");
  }
  const savedSize = node.size?.length >= 2 ? [Number(node.size[0]), Number(node.size[1])] : null;
  beforeRemove?.(currentWidget);
  node.removeWidget(currentWidget);
  const nextWidget = createWidget();
  if (savedSize) {
    if (typeof node.setSize === "function") {
      node.setSize(savedSize);
    } else {
      node.size = savedSize;
    }
  }
  node._widgetSlotsDirty = true;
  return nextWidget;
}

export function isClientPointInsideRect({ clientX, clientY, rect }) {
  const left = Number(rect?.left);
  const top = Number(rect?.top);
  const width = Number(rect?.width);
  const height = Number(rect?.height);
  if (!Number.isFinite(left) || !Number.isFinite(top) || !(width > 0) || !(height > 0)) {
    return false;
  }
  return clientX >= left && clientX <= left + width && clientY >= top && clientY <= top + height;
}

function normalizeImageOptionQuery(query) {
  return String(query || "").trim().toLowerCase();
}

export function filterImageOptions({ nextOptions = [], query = "" }) {
  const normalizedOptions = Array.isArray(nextOptions) ? nextOptions.map((value) => String(value)) : [];
  const normalizedQuery = normalizeImageOptionQuery(query);
  if (!normalizedQuery) {
    return normalizedOptions;
  }
  return normalizedOptions.filter((value) => value.toLowerCase().includes(normalizedQuery));
}

export function selectSyncedImageValue({ preferredValue = "", currentValue = "", nextOptions = [] }) {
  const normalizedOptions = Array.isArray(nextOptions) ? nextOptions.map((value) => String(value)) : [];
  const preferred = String(preferredValue || "");
  const current = String(currentValue || "");

  if (preferred && normalizedOptions.includes(preferred)) {
    return preferred;
  }
  if (current && normalizedOptions.includes(current)) {
    return current;
  }
  return normalizedOptions[0] || "";
}

export function sameStringArray(left, right) {
  if (left === right) {
    return true;
  }
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    if (String(left[index]) !== String(right[index])) {
      return false;
    }
  }
  return true;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export const BUCKET_BOUNDARY_ALPHA = Math.fround(1 - 1e-6);

export function maskAlphaToGrayscale(alpha) {
  return Math.round(clamp(Number(alpha) || 0, 0, 1) * 255);
}

function isBucketBoundary(alpha) {
  return alpha >= BUCKET_BOUNDARY_ALPHA;
}

function normalizeMaskBounds({ imageWidth, imageHeight, crop }) {
  const width = Math.max(1, Math.floor(Number(imageWidth) || 1));
  const height = Math.max(1, Math.floor(Number(imageHeight) || 1));
  const left = clamp(Math.floor(Number(crop?.x) || 0), 0, width);
  const top = clamp(Math.floor(Number(crop?.y) || 0), 0, height);
  const right = clamp(Math.ceil((Number(crop?.x) || 0) + (Number(crop?.width) || 0)), left, width);
  const bottom = clamp(Math.ceil((Number(crop?.y) || 0) + (Number(crop?.height) || 0)), top, height);
  return { width, height, left, top, right, bottom };
}

function getSourceBrushAlpha(distance, radius, softness, opacity) {
  if (radius <= 0 || distance > radius) {
    return 0;
  }
  const resolvedSoftness = clamp(Number(softness) || 0, 0, 1);
  const resolvedOpacity = clamp(Number(opacity) || 0, 0, 1);
  if (resolvedSoftness <= 0) {
    return resolvedOpacity;
  }
  const coreRadius = radius * Math.max(0, 1 - resolvedSoftness);
  if (distance <= coreRadius) {
    return resolvedOpacity;
  }
  const feather = Math.max(radius - coreRadius, 0.000001);
  const normalized = 1 - (distance - coreRadius) / feather;
  return resolvedOpacity * clamp(normalized, 0, 1) ** 2;
}

function applySourceBrushDab(mask, bounds, centerX, centerY, radius, softness, opacity, mode) {
  const left = Math.max(Math.floor(centerX - radius), bounds.left);
  const top = Math.max(Math.floor(centerY - radius), bounds.top);
  const right = Math.min(Math.ceil(centerX + radius) + 1, bounds.right);
  const bottom = Math.min(Math.ceil(centerY + radius) + 1, bounds.bottom);

  for (let y = top; y < bottom; y += 1) {
    const rowOffset = y * bounds.width;
    for (let x = left; x < right; x += 1) {
      const alpha = getSourceBrushAlpha(Math.hypot(x - centerX, y - centerY), radius, softness, opacity);
      if (alpha <= 0) {
        continue;
      }
      const index = rowOffset + x;
      if (mode === "erase") {
        mask[index] *= 1 - alpha;
      } else {
        mask[index] = Math.max(mask[index], alpha);
      }
    }
  }
}

function applySourceRuns(mask, bounds, runs) {
  if (!Array.isArray(runs)) {
    return;
  }
  for (const run of runs) {
    if (!Array.isArray(run) || run.length !== 3) {
      continue;
    }
    const y = Math.round(Number(run[0]));
    if (y < bounds.top || y >= bounds.bottom) {
      continue;
    }
    const left = clamp(Math.round(Number(run[1])), bounds.left, bounds.right);
    const right = clamp(Math.round(Number(run[2])), left, bounds.right);
    if (right > left) {
      mask.fill(1, y * bounds.width + left, y * bounds.width + right);
    }
  }
}

export function renderSourceMask({ imageWidth, imageHeight, crop, operations = [] }) {
  const bounds = normalizeMaskBounds({ imageWidth, imageHeight, crop });
  const mask = new Float32Array(bounds.width * bounds.height);

  for (const operation of Array.isArray(operations) ? operations : []) {
    const mode = operation?.mode === "erase" ? "erase" : operation?.mode;
    if (mode === "fill") {
      for (let y = bounds.top; y < bounds.bottom; y += 1) {
        mask.fill(1, y * bounds.width + bounds.left, y * bounds.width + bounds.right);
      }
      continue;
    }
    if (mode === "bucket") {
      applySourceRuns(mask, bounds, operation.runs);
      continue;
    }

    const points = Array.isArray(operation?.points) ? operation.points : [];
    if (!points.length) {
      continue;
    }
    const radiusValue = Number(operation.radius_px);
    const softnessValue = Number(operation.softness);
    const opacityValue = Number(operation.opacity);
    const radius = Math.max(Number.isFinite(radiusValue) ? radiusValue : 48, 1);
    const softness = clamp(Number.isFinite(softnessValue) ? softnessValue : 0.35, 0, 1);
    const opacity = clamp(Number.isFinite(opacityValue) ? opacityValue : 1, 0, 1);
    const applyPoint = (point) => {
      applySourceBrushDab(
        mask,
        bounds,
        Number(point?.[0]) || 0,
        Number(point?.[1]) || 0,
        radius,
        softness,
        opacity,
        mode
      );
    };

    if (points.length === 1) {
      applyPoint(points[0]);
      continue;
    }

    const step = Math.max(radius * 0.25, 1);
    for (let index = 0; index < points.length - 1; index += 1) {
      const start = points[index];
      const end = points[index + 1];
      const startX = Number(start?.[0]) || 0;
      const startY = Number(start?.[1]) || 0;
      const dx = (Number(end?.[0]) || 0) - startX;
      const dy = (Number(end?.[1]) || 0) - startY;
      const distance = Math.hypot(dx, dy);
      const steps = Math.max(Math.ceil(distance / step), 1);
      for (let stepIndex = 0; stepIndex <= steps; stepIndex += 1) {
        const t = stepIndex / steps;
        applySourceBrushDab(
          mask,
          bounds,
          startX + dx * t,
          startY + dy * t,
          radius,
          softness,
          opacity,
          mode
        );
      }
    }
  }

  return mask;
}

export function buildBucketFillRuns({ mask, imageWidth, imageHeight, crop, seedX, seedY }) {
  const bounds = normalizeMaskBounds({ imageWidth, imageHeight, crop });
  if (!(mask instanceof Float32Array) || mask.length !== bounds.width * bounds.height) {
    throw new Error("mask dimensions do not match the source image");
  }

  const startX = Math.floor(Number(seedX));
  const startY = Math.floor(Number(seedY));
  if (startX < bounds.left || startX >= bounds.right || startY < bounds.top || startY >= bounds.bottom) {
    return { runs: [], pixelCount: 0, reason: "outside" };
  }
  if (isBucketBoundary(mask[startY * bounds.width + startX])) {
    return { runs: [], pixelCount: 0, reason: "masked" };
  }

  const stack = [startX, startY];
  const runs = [];
  let pixelCount = 0;

  while (stack.length) {
    const y = stack.pop();
    const x = stack.pop();
    const rowOffset = y * bounds.width;
    if (isBucketBoundary(mask[rowOffset + x])) {
      continue;
    }

    let left = x;
    while (left > bounds.left && !isBucketBoundary(mask[rowOffset + left - 1])) {
      left -= 1;
    }
    let right = x;
    while (right + 1 < bounds.right && !isBucketBoundary(mask[rowOffset + right + 1])) {
      right += 1;
    }

    mask.fill(1, rowOffset + left, rowOffset + right + 1);
    runs.push([y, left, right + 1]);
    pixelCount += right - left + 1;

    for (const adjacentY of [y - 1, y + 1]) {
      if (adjacentY < bounds.top || adjacentY >= bounds.bottom) {
        continue;
      }
      const adjacentOffset = adjacentY * bounds.width;
      let scanX = left;
      while (scanX <= right) {
        while (scanX <= right && isBucketBoundary(mask[adjacentOffset + scanX])) {
          scanX += 1;
        }
        if (scanX > right) {
          break;
        }
        stack.push(scanX, adjacentY);
        while (scanX <= right && !isBucketBoundary(mask[adjacentOffset + scanX])) {
          scanX += 1;
        }
      }
    }
  }

  runs.sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  return { runs, pixelCount, reason: null };
}
