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
