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
