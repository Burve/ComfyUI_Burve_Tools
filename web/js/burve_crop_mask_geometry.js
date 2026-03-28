function toFiniteNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function toPositiveNumber(value, fallback = 1) {
  return Math.max(toFiniteNumber(value, fallback), 0.0001);
}

function normalizeRect(rect, logicalWidth, logicalHeight) {
  return {
    left: toFiniteNumber(rect?.left, 0),
    top: toFiniteNumber(rect?.top, 0),
    width: toPositiveNumber(rect?.width, logicalWidth),
    height: toPositiveNumber(rect?.height, logicalHeight),
  };
}

export function getLogicalCanvasMetrics({ canvas, logicalWidth, logicalHeight }) {
  const resolvedLogicalWidth = toPositiveNumber(logicalWidth, canvas?.clientWidth || 1);
  const resolvedLogicalHeight = toPositiveNumber(logicalHeight, canvas?.clientHeight || 1);
  const rect = normalizeRect(canvas?.getBoundingClientRect?.(), resolvedLogicalWidth, resolvedLogicalHeight);

  return {
    rect,
    logicalWidth: resolvedLogicalWidth,
    logicalHeight: resolvedLogicalHeight,
    scaleX: resolvedLogicalWidth / rect.width,
    scaleY: resolvedLogicalHeight / rect.height,
  };
}

export function clientToLogicalCanvasPoint({ clientX, clientY, rect, logicalWidth, logicalHeight }) {
  const normalizedRect = normalizeRect(rect, logicalWidth, logicalHeight);
  const resolvedLogicalWidth = toPositiveNumber(logicalWidth, normalizedRect.width);
  const resolvedLogicalHeight = toPositiveNumber(logicalHeight, normalizedRect.height);

  return {
    x: (toFiniteNumber(clientX, normalizedRect.left) - normalizedRect.left) * (resolvedLogicalWidth / normalizedRect.width),
    y: (toFiniteNumber(clientY, normalizedRect.top) - normalizedRect.top) * (resolvedLogicalHeight / normalizedRect.height),
  };
}

export function logicalToClientCanvasPoint({ x, y, rect, logicalWidth, logicalHeight }) {
  const normalizedRect = normalizeRect(rect, logicalWidth, logicalHeight);
  const resolvedLogicalWidth = toPositiveNumber(logicalWidth, normalizedRect.width);
  const resolvedLogicalHeight = toPositiveNumber(logicalHeight, normalizedRect.height);

  return {
    x: normalizedRect.left + toFiniteNumber(x, 0) * (normalizedRect.width / resolvedLogicalWidth),
    y: normalizedRect.top + toFiniteNumber(y, 0) * (normalizedRect.height / resolvedLogicalHeight),
  };
}

export function computeZoomAnchoredViewport({
  stageWidth,
  stageHeight,
  imageWidth,
  imageHeight,
  fitScale,
  nextZoom,
  anchorCanvasPoint,
  anchorImagePoint,
}) {
  const scale = toPositiveNumber(fitScale, 0.0001) * toPositiveNumber(nextZoom, 1);
  const resolvedStageWidth = toFiniteNumber(stageWidth, imageWidth);
  const resolvedStageHeight = toFiniteNumber(stageHeight, imageHeight);
  const resolvedImageWidth = toFiniteNumber(imageWidth, 0);
  const resolvedImageHeight = toFiniteNumber(imageHeight, 0);
  const anchorX = toFiniteNumber(anchorCanvasPoint?.x, resolvedStageWidth / 2);
  const anchorY = toFiniteNumber(anchorCanvasPoint?.y, resolvedStageHeight / 2);
  const imageX = toFiniteNumber(anchorImagePoint?.x, resolvedImageWidth / 2);
  const imageY = toFiniteNumber(anchorImagePoint?.y, resolvedImageHeight / 2);
  const left = (resolvedStageWidth - resolvedImageWidth * scale) / 2;
  const top = (resolvedStageHeight - resolvedImageHeight * scale) / 2;

  return {
    zoom: toPositiveNumber(nextZoom, 1),
    pan_x: (anchorX - left - imageX * scale) / scale,
    pan_y: (anchorY - top - imageY * scale) / scale,
  };
}
