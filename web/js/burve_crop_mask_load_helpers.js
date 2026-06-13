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

export function computeCropMaskWidgetLayout({ widgetHeight, chromeHeight, minCanvasHeight }) {
  const nextWidgetHeight = Math.max(widgetHeight, Math.ceil(chromeHeight + minCanvasHeight));
  return {
    widgetHeight: nextWidgetHeight,
    canvasHeight: Math.max(1, Math.floor(nextWidgetHeight - chromeHeight)),
  };
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
