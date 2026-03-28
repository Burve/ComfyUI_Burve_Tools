import { app } from "../../../scripts/app.js";
import {
  clientToLogicalCanvasPoint,
  computeZoomAnchoredViewport,
  getLogicalCanvasMetrics,
} from "./burve_crop_mask_geometry.js";

const TARGET_NODE = "BurveCropMaskLoad";
const INTERNAL_WIDGET = "editor_state_json";
const CANVAS_MIN_HEIGHT = 420;
const STYLE_ID = "burve-crop-mask-load-style";
const DEFAULT_BRUSH_RADIUS = 48;
const DEFAULT_BRUSH_SOFTNESS = 0.35;
const DEFAULT_BRUSH_OPACITY = 1.0;
const HANDLE_HIT_RADIUS = 16;
const HANDLE_DRAW_RADIUS = 5;
const PAINT_STROKE = "rgba(160, 245, 255, 0.94)";
const ERASE_STROKE = "rgba(255, 255, 255, 0.94)";
const CROP_STROKE = "rgba(220, 191, 122, 0.95)";
const CROP_GRID = "rgba(220, 191, 122, 0.24)";
const PAINT_RGBA = [56, 219, 255];
const PAINT_MAX_ALPHA = 0.34;

function injectStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .burve-crop-root {
      display: flex;
      flex-direction: column;
      gap: 10px;
      width: 100%;
      height: auto;
      min-height: 0;
      box-sizing: border-box;
      padding: 10px;
      border-radius: 14px;
      background:
        radial-gradient(circle at 10% 10%, rgba(58, 77, 101, 0.22), transparent 42%),
        linear-gradient(180deg, rgba(26, 28, 32, 0.97), rgba(13, 15, 18, 0.97));
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 18px 44px rgba(0, 0, 0, 0.22);
      color: #f4efe6;
      font-family: Georgia, "Times New Roman", serif;
    }
    .burve-crop-toolbar,
    .burve-crop-status {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .burve-crop-title {
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #dcbf7a;
      margin-right: auto;
    }
    .burve-crop-toolbar button {
      border: 1px solid rgba(220, 191, 122, 0.28);
      background: rgba(255, 255, 255, 0.04);
      color: #f4efe6;
      border-radius: 999px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }
    .burve-crop-toolbar button.active {
      background: rgba(220, 191, 122, 0.18);
      border-color: rgba(220, 191, 122, 0.65);
      box-shadow: 0 0 0 1px rgba(220, 191, 122, 0.2) inset;
    }
    .burve-crop-toolbar label {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      font-size: 11px;
      color: #d7dce4;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.04);
    }
    .burve-crop-toolbar input[type="range"] {
      width: 88px;
    }
    .burve-crop-value {
      display: inline-flex;
      min-width: 40px;
      justify-content: flex-end;
      color: #9ef5ff;
      font-variant-numeric: tabular-nums;
    }
    .burve-crop-canvas-wrap {
      position: relative;
      flex: 0 0 auto;
      width: 100%;
      height: ${CANVAS_MIN_HEIGHT}px;
      min-height: ${CANVAS_MIN_HEIGHT}px;
      border-radius: 12px;
      overflow: hidden;
      background:
        radial-gradient(circle at 20% 20%, rgba(63, 77, 97, 0.18), transparent 45%),
        linear-gradient(135deg, rgba(11, 14, 18, 0.95), rgba(24, 28, 35, 0.96));
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .burve-crop-canvas {
      width: 100%;
      height: 100%;
      display: block;
      touch-action: none;
      cursor: crosshair;
    }
    .burve-crop-status {
      justify-content: space-between;
      font-size: 11px;
      color: #b9c0cb;
    }
  `;
  document.head.appendChild(style);
}

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function parseRatio(ratio) {
  const [w, h] = String(ratio || "1:1").split(":").map((value) => Number.parseInt(value, 10));
  return { w: w || 1, h: h || 1 };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function buildDefaultCrop(imageWidth, imageHeight, ratioValue) {
  const ratio = parseRatio(ratioValue);
  let width;
  let height;

  if (imageWidth * ratio.h >= imageHeight * ratio.w) {
    height = imageHeight;
    width = Math.round((height * ratio.w) / ratio.h);
  } else {
    width = imageWidth;
    height = Math.round((width * ratio.h) / ratio.w);
  }

  width = Math.max(1, Math.min(width, imageWidth));
  height = Math.max(1, Math.min(height, imageHeight));
  return {
    x: Math.max(0, Math.floor((imageWidth - width) / 2)),
    y: Math.max(0, Math.floor((imageHeight - height) / 2)),
    width,
    height,
  };
}

function fitCropNearCenter(crop, imageWidth, imageHeight, ratioValue) {
  const ratio = parseRatio(ratioValue);
  const centerX = (crop.x ?? 0) + (crop.width ?? imageWidth) / 2;
  const centerY = (crop.y ?? 0) + (crop.height ?? imageHeight) / 2;
  let width = Math.max(1, Math.min(crop.width ?? imageWidth, imageWidth));
  let height = Math.round((width * ratio.h) / ratio.w);

  if (height > imageHeight) {
    height = imageHeight;
    width = Math.round((height * ratio.w) / ratio.h);
  }

  let x = centerX - width / 2;
  let y = centerY - height / 2;
  x = clamp(x, 0, imageWidth - width);
  y = clamp(y, 0, imageHeight - height);

  return {
    x: Math.round(x),
    y: Math.round(y),
    width: Math.max(1, Math.round(width)),
    height: Math.max(1, Math.round(height)),
  };
}

function normalizePointDistance(lastPoint, point) {
  if (!lastPoint) {
    return Number.POSITIVE_INFINITY;
  }
  const dx = point[0] - lastPoint[0];
  const dy = point[1] - lastPoint[1];
  return Math.hypot(dx, dy);
}

function buildViewUrl(imageValue) {
  const normalized = String(imageValue || "").replace(/\\/g, "/");
  const slash = normalized.lastIndexOf("/");
  const filename = slash >= 0 ? normalized.slice(slash + 1) : normalized;
  const subfolder = slash >= 0 ? normalized.slice(0, slash) : "";
  const params = new URLSearchParams({ filename, type: "input" });
  if (subfolder) {
    params.set("subfolder", subfolder);
  }
  return `/view?${params.toString()}`;
}

function createHandleMap(cropRect) {
  const left = cropRect.left;
  const top = cropRect.top;
  const right = cropRect.left + cropRect.width;
  const bottom = cropRect.top + cropRect.height;
  const centerX = left + cropRect.width / 2;
  const centerY = top + cropRect.height / 2;
  return {
    nw: { x: left, y: top },
    n: { x: centerX, y: top },
    ne: { x: right, y: top },
    e: { x: right, y: centerY },
    se: { x: right, y: bottom },
    s: { x: centerX, y: bottom },
    sw: { x: left, y: bottom },
    w: { x: left, y: centerY },
  };
}

function cursorForHandle(handle) {
  switch (handle) {
    case "n":
    case "s":
      return "ns-resize";
    case "e":
    case "w":
      return "ew-resize";
    case "ne":
    case "sw":
      return "nesw-resize";
    case "nw":
    case "se":
      return "nwse-resize";
    default:
      return "grab";
  }
}

function getBrushAlpha(distance, radius, softness, opacity = 1) {
  if (radius <= 0 || distance >= radius) {
    return 0;
  }

  const clampedSoftness = clamp(softness ?? DEFAULT_BRUSH_SOFTNESS, 0, 1);
  const clampedOpacity = clamp(opacity ?? DEFAULT_BRUSH_OPACITY, 0, 1);
  if (clampedSoftness <= 0.001) {
    return clampedOpacity;
  }

  const innerRadius = radius * Math.max(0.05, 1 - clampedSoftness);
  if (distance <= innerRadius) {
    return clampedOpacity;
  }

  const normalized = 1 - (distance - innerRadius) / Math.max(0.0001, radius - innerRadius);
  const falloff = clamp(normalized, 0, 1) ** 2;
  return clampedOpacity * falloff;
}

function applyBrushDabToAlphaBuffer(alphaBuffer, bufferWidth, centerX, centerY, radius, softness, opacity, mode, clipRect) {
  if (!clipRect || radius <= 0 || clipRect.right <= clipRect.left || clipRect.bottom <= clipRect.top) {
    return;
  }

  const left = clamp(Math.floor(centerX - radius - 1), clipRect.left, clipRect.right - 1);
  const top = clamp(Math.floor(centerY - radius - 1), clipRect.top, clipRect.bottom - 1);
  const right = clamp(Math.ceil(centerX + radius + 1), clipRect.left, clipRect.right - 1);
  const bottom = clamp(Math.ceil(centerY + radius + 1), clipRect.top, clipRect.bottom - 1);

  for (let y = top; y <= bottom; y += 1) {
    const py = y + 0.5;
    const rowOffset = y * bufferWidth;
    for (let x = left; x <= right; x += 1) {
      const px = x + 0.5;
      const alpha = getBrushAlpha(Math.hypot(px - centerX, py - centerY), radius, softness, opacity);
      if (alpha <= 0) {
        continue;
      }
      const index = rowOffset + x;
      if (mode === "erase") {
        alphaBuffer[index] *= 1 - alpha;
      } else {
        alphaBuffer[index] = Math.max(alphaBuffer[index], alpha);
      }
    }
  }
}

class CropMaskEditor {
  constructor(node) {
    this.node = node;
    this.imageWidget = getWidget(node, "image");
    this.ratioWidget = getWidget(node, "aspect_ratio");
    this.stateWidget = getWidget(node, INTERNAL_WIDGET);
    this.sourceImage = null;
    this.state = null;
    this.pointerMode = null;
    this.dragHandle = null;
    this.dragStart = null;
    this.activeStroke = null;
    this.hoverPointer = null;
    this.hoverImagePoint = null;
    this.hoverHandle = null;
    this.spacePressed = false;
    this.saveTimer = null;
    this.lastWidgetImage = null;
    this.lastWidgetRatio = null;
    this.lastLoadedUrl = null;
    injectStyles();
    this.root = document.createElement("div");
    this.root.className = "burve-crop-root";
    this.root.innerHTML = `
      <div class="burve-crop-toolbar">
        <div class="burve-crop-title">Burve Crop Lab</div>
        <button data-tool="crop" class="active">Crop</button>
        <button data-tool="paint">Paint</button>
        <button data-tool="erase">Erase</button>
        <label>Brush Size <input data-role="brush-size" type="range" min="4" max="256" step="1" value="${DEFAULT_BRUSH_RADIUS}" /><span class="burve-crop-value" data-role="brush-value">${DEFAULT_BRUSH_RADIUS}px</span></label>
        <label>Softness <input data-role="softness" type="range" min="0" max="1" step="0.01" value="${DEFAULT_BRUSH_SOFTNESS}" /></label>
        <button data-action="fit">Fit</button>
        <button data-action="reset-crop">Reset Crop</button>
        <button data-action="clear-mask">Clear Mask</button>
      </div>
      <div class="burve-crop-canvas-wrap">
        <canvas class="burve-crop-canvas"></canvas>
      </div>
      <div class="burve-crop-status">
        <span data-role="crop-info">No image selected</span>
        <span data-role="hint">Wheel: zoom | Space/middle drag: pan</span>
      </div>
    `;

    this.canvasWrap = this.root.querySelector(".burve-crop-canvas-wrap");
    this.canvas = this.root.querySelector(".burve-crop-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.overlayCanvas = document.createElement("canvas");
    this.overlayCtx = this.overlayCanvas.getContext("2d");
    this.toolbar = this.root.querySelector(".burve-crop-toolbar");
    this.statusBar = this.root.querySelector(".burve-crop-status");
    this.cropInfo = this.root.querySelector('[data-role="crop-info"]');
    this.brushSizeInput = this.root.querySelector('[data-role="brush-size"]');
    this.brushValue = this.root.querySelector('[data-role="brush-value"]');
    this.softnessInput = this.root.querySelector('[data-role="softness"]');
    this.toolButtons = [...this.root.querySelectorAll("[data-tool]")];
    this.actionButtons = [...this.root.querySelectorAll("[data-action]")];
    this.tool = "crop";
    this.widgetHeight = CANVAS_MIN_HEIGHT + 180;
    this.stageMetrics = {
      cssWidth: 320,
      cssHeight: CANVAS_MIN_HEIGHT,
      dpr: 1,
      canvasRect: null,
      clientToLogicalScaleX: 1,
      clientToLogicalScaleY: 1,
      toolbarHeight: 0,
      statusHeight: 0,
      paddingTop: 0,
      paddingBottom: 0,
      gap: 10,
    };

    this.toolButtons.forEach((button) => {
      button.addEventListener("click", () => this.setTool(button.dataset.tool));
    });
    this.actionButtons.forEach((button) => {
      button.addEventListener("click", () => this.handleAction(button.dataset.action));
    });
    this.brushSizeInput.addEventListener("input", () => {
      this.ensureState();
      this.state.brush.radius_px = Number(this.brushSizeInput.value);
      this.updateBrushValue();
      this.scheduleSave();
      this.draw();
    });
    this.softnessInput.addEventListener("input", () => {
      this.ensureState();
      this.state.brush.softness = Number(this.softnessInput.value);
      this.scheduleSave();
      this.draw();
    });

    this.canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
    this.canvas.addEventListener("pointerup", (event) => this.onPointerUp(event));
    this.canvas.addEventListener("pointercancel", (event) => this.onPointerUp(event));
    this.canvas.addEventListener("pointerleave", () => this.onPointerLeave());
    this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);

    this.domWidget = node.addDOMWidget("burve_crop_mask_editor", "BURVE_CROP_MASK_EDITOR", this.root, {
      hideOnZoom: false,
      getHeight: () => this.widgetHeight,
      getMinHeight: () => this.widgetHeight,
      getValue: () => "",
      setValue: () => {},
      afterResize: () => this.draw(),
    });
    this.domWidget.serialize = false;
    this.hideStateWidget();
    this.installWidgetCallbacks();
    this.updateBrushValue();
    this.refreshFromWidgets(true);
    this.updateNodeSize();
  }

  onKeyDown = (event) => {
    if (event.code === "Space") {
      this.spacePressed = true;
      this.updateCursor();
    }
  };

  onKeyUp = (event) => {
    if (event.code === "Space") {
      this.spacePressed = false;
      this.updateCursor();
    }
  };

  destroy() {
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
  }

  updateNodeSize() {
    this.node.size = [
      Math.max(this.node.size?.[0] ?? 0, 520),
      Math.max(this.node.size?.[1] ?? 0, Math.ceil(this.widgetHeight + 48)),
    ];
  }

  hideStateWidget() {
    if (!this.stateWidget) {
      return;
    }
    this.stateWidget.hidden = true;
    this.stateWidget.computeSize = () => [0, -4];
  }

  installWidgetCallbacks() {
    [this.imageWidget, this.ratioWidget].forEach((widget) => {
      if (!widget) {
        return;
      }
      const previous = widget.callback;
      widget.callback = (...args) => {
        const result = previous?.apply(widget, args);
        this.refreshFromWidgets();
        return result;
      };
    });
  }

  updateBrushValue() {
    if (this.brushValue) {
      this.brushValue.textContent = `${Number(this.brushSizeInput.value)}px`;
    }
  }

  setTool(tool) {
    this.tool = tool;
    this.ensureState();
    if (this.state && tool !== "crop") {
      this.state.brush.mode = tool === "erase" ? "erase" : "paint";
    }
    this.toolButtons.forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === tool);
    });
    this.updateCursor();
    this.scheduleSave();
    this.draw();
  }

  handleAction(action) {
    this.ensureState();
    if (!this.sourceImage) {
      return;
    }
    if (action === "fit") {
      this.state.viewport = { zoom: 1, pan_x: 0, pan_y: 0 };
    } else if (action === "reset-crop") {
      this.state.crop = buildDefaultCrop(this.sourceImage.width, this.sourceImage.height, this.getRatioValue());
    } else if (action === "clear-mask") {
      this.state.strokes = [];
    }
    this.scheduleSave();
    this.draw();
  }

  getRatioValue() {
    return this.ratioWidget?.value || "1:1";
  }

  getImageValue() {
    return this.imageWidget?.value || "";
  }

  readStoredState() {
    if (!this.stateWidget?.value) {
      return null;
    }
    try {
      const parsed = JSON.parse(this.stateWidget.value);
      return parsed && typeof parsed === "object" ? parsed : null;
    } catch {
      return null;
    }
  }

  ensureState() {
    if (this.state || !this.sourceImage) {
      return;
    }

    const stored = this.readStoredState();
    const imageName = this.getImageValue();
    const ratioValue = this.getRatioValue();
    const defaultState = {
      version: 1,
      image_name: imageName,
      source_size: { width: this.sourceImage.width, height: this.sourceImage.height },
      aspect_ratio: ratioValue,
      crop: buildDefaultCrop(this.sourceImage.width, this.sourceImage.height, ratioValue),
      viewport: { zoom: 1, pan_x: 0, pan_y: 0 },
      brush: {
        radius_px: DEFAULT_BRUSH_RADIUS,
        softness: DEFAULT_BRUSH_SOFTNESS,
        opacity: DEFAULT_BRUSH_OPACITY,
        mode: "paint",
      },
      strokes: [],
    };

    if (
      !stored ||
      stored.image_name !== imageName ||
      stored?.source_size?.width !== this.sourceImage.width ||
      stored?.source_size?.height !== this.sourceImage.height
    ) {
      this.state = defaultState;
    } else {
      this.state = {
        ...defaultState,
        ...stored,
        crop: stored.crop
          ? fitCropNearCenter(stored.crop, this.sourceImage.width, this.sourceImage.height, ratioValue)
          : defaultState.crop,
        viewport: { ...defaultState.viewport, ...(stored.viewport || {}) },
        brush: { ...defaultState.brush, ...(stored.brush || {}) },
        strokes: Array.isArray(stored.strokes) ? stored.strokes : [],
      };
      this.state.aspect_ratio = ratioValue;
    }

    this.brushSizeInput.value = String(this.state.brush.radius_px ?? DEFAULT_BRUSH_RADIUS);
    this.softnessInput.value = String(this.state.brush.softness ?? DEFAULT_BRUSH_SOFTNESS);
    this.updateBrushValue();
    this.scheduleSave();
  }

  refreshFromWidgets(force = false) {
    const imageValue = this.getImageValue();
    const ratioValue = this.getRatioValue();
    if (!force && imageValue === this.lastWidgetImage && ratioValue === this.lastWidgetRatio) {
      return;
    }
    this.lastWidgetImage = imageValue;
    this.lastWidgetRatio = ratioValue;
    this.loadPreviewImage(imageValue, ratioValue);
  }

  loadPreviewImage(imageValue, ratioValue) {
    if (!imageValue) {
      this.sourceImage = null;
      this.state = null;
      this.hoverPointer = null;
      this.hoverImagePoint = null;
      this.hoverHandle = null;
      this.draw();
      return;
    }

    const nextUrl = buildViewUrl(imageValue);
    if (nextUrl === this.lastLoadedUrl && this.sourceImage) {
      if (this.state) {
        this.state.crop = fitCropNearCenter(this.state.crop, this.sourceImage.width, this.sourceImage.height, ratioValue);
        this.state.aspect_ratio = ratioValue;
      }
      this.draw();
      return;
    }

    this.lastLoadedUrl = nextUrl;
    const image = new Image();
    image.onload = () => {
      this.sourceImage = image;
      this.state = null;
      this.hoverPointer = null;
      this.hoverImagePoint = null;
      this.hoverHandle = null;
      this.ensureState();
      if (this.state) {
        this.state.crop = fitCropNearCenter(this.state.crop, this.sourceImage.width, this.sourceImage.height, ratioValue);
        this.state.aspect_ratio = ratioValue;
      }
      this.draw();
    };
    image.onerror = () => {
      this.sourceImage = null;
      this.state = null;
      this.draw();
    };
    image.src = nextUrl;
  }

  scheduleSave() {
    if (!this.stateWidget || !this.state) {
      return;
    }
    window.clearTimeout(this.saveTimer);
    this.saveTimer = window.setTimeout(() => {
      this.stateWidget.value = JSON.stringify(this.state);
      app.graph?.setDirtyCanvas?.(true, true);
    }, 60);
  }

  layoutStage() {
    const rootStyles = window.getComputedStyle(this.root);
    const paddingTop = Number.parseFloat(rootStyles.paddingTop || "0") || 0;
    const paddingBottom = Number.parseFloat(rootStyles.paddingBottom || "0") || 0;
    const gap = Number.parseFloat(rootStyles.rowGap || rootStyles.gap || "10") || 10;
    const toolbarHeight = Math.ceil(this.toolbar?.offsetHeight || this.toolbar?.clientHeight || 0);
    const statusHeight = Math.ceil(this.statusBar?.offsetHeight || this.statusBar?.clientHeight || 0);
    const chromeHeight = paddingTop + paddingBottom + toolbarHeight + statusHeight + gap * 2;
    const hostHeight = Math.max(1, Math.floor(this.root.parentElement?.clientHeight || this.widgetHeight));
    const minimumWidgetHeight = Math.ceil(chromeHeight + CANVAS_MIN_HEIGHT);

    if (this.widgetHeight < minimumWidgetHeight) {
      this.widgetHeight = minimumWidgetHeight;
      this.updateNodeSize();
      app.graph?.setDirtyCanvas?.(true, true);
    }

    this.root.style.height = `${hostHeight}px`;
    const stageHeight = Math.max(1, Math.floor(hostHeight - chromeHeight));
    this.canvasWrap.style.height = `${stageHeight}px`;
    this.canvasWrap.style.minHeight = `${stageHeight}px`;

    const cssWidth = Math.max(320, Math.floor(this.canvasWrap.clientWidth || this.stageMetrics.cssWidth || 320));
    const cssHeight = Math.max(1, Math.floor(this.canvasWrap.clientHeight || stageHeight || this.stageMetrics.cssHeight || CANVAS_MIN_HEIGHT));
    const dpr = Math.max(window.devicePixelRatio || 1, 1);
    const targetWidth = Math.max(1, Math.round(cssWidth * dpr));
    const targetHeight = Math.max(1, Math.round(cssHeight * dpr));

    if (this.canvas.width !== targetWidth || this.canvas.height !== targetHeight) {
      this.canvas.width = targetWidth;
      this.canvas.height = targetHeight;
    }
    if (this.overlayCanvas.width !== targetWidth || this.overlayCanvas.height !== targetHeight) {
      this.overlayCanvas.width = targetWidth;
      this.overlayCanvas.height = targetHeight;
    }

    this.canvas.style.width = `${cssWidth}px`;
    this.canvas.style.height = `${cssHeight}px`;
    this.overlayCtx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const canvasMetrics = getLogicalCanvasMetrics({
      canvas: this.canvas,
      logicalWidth: cssWidth,
      logicalHeight: cssHeight,
    });
    this.stageMetrics = {
      cssWidth: canvasMetrics.logicalWidth,
      cssHeight: canvasMetrics.logicalHeight,
      dpr,
      canvasRect: canvasMetrics.rect,
      clientToLogicalScaleX: canvasMetrics.scaleX,
      clientToLogicalScaleY: canvasMetrics.scaleY,
      toolbarHeight,
      statusHeight,
      paddingTop,
      paddingBottom,
      gap,
    };
  }

  getViewTransform() {
    const imageWidth = this.sourceImage.width;
    const imageHeight = this.sourceImage.height;
    const padding = 20;
    const fitScale = Math.max(
      0.0001,
      Math.min(
        Math.max(1, this.stageMetrics.cssWidth - padding * 2) / imageWidth,
        Math.max(1, this.stageMetrics.cssHeight - padding * 2) / imageHeight
      )
    );
    const scale = fitScale * (this.state?.viewport?.zoom || 1);
    const left = (this.stageMetrics.cssWidth - imageWidth * scale) / 2 + (this.state?.viewport?.pan_x || 0) * scale;
    const top = (this.stageMetrics.cssHeight - imageHeight * scale) / 2 + (this.state?.viewport?.pan_y || 0) * scale;
    return {
      fitScale,
      scale,
      left,
      top,
      imageRectOnCanvas: {
        left,
        top,
        width: imageWidth * scale,
        height: imageHeight * scale,
      },
    };
  }

  getImageRect(transform) {
    return {
      left: transform.left,
      top: transform.top,
      width: this.sourceImage.width * transform.scale,
      height: this.sourceImage.height * transform.scale,
    };
  }

  imageToScreen(point, transform = this.getViewTransform()) {
    return {
      x: transform.left + point.x * transform.scale,
      y: transform.top + point.y * transform.scale,
    };
  }

  screenToImage(point, transform = this.getViewTransform()) {
    return {
      x: (point.x - transform.left) / transform.scale,
      y: (point.y - transform.top) / transform.scale,
    };
  }

  getPointerPoint(event) {
    const bounds =
      this.stageMetrics.canvasRect ||
      getLogicalCanvasMetrics({
        canvas: this.canvas,
        logicalWidth: this.stageMetrics.cssWidth,
        logicalHeight: this.stageMetrics.cssHeight,
      }).rect;
    return clientToLogicalCanvasPoint({
      clientX: event.clientX,
      clientY: event.clientY,
      rect: bounds,
      logicalWidth: this.stageMetrics.cssWidth,
      logicalHeight: this.stageMetrics.cssHeight,
    });
  }

  isPointInsideImage(point) {
    if (!this.sourceImage) {
      return false;
    }
    return point.x >= 0 && point.y >= 0 && point.x <= this.sourceImage.width && point.y <= this.sourceImage.height;
  }

  isPointInsideCrop(point) {
    if (!this.state?.crop) {
      return false;
    }
    const crop = this.state.crop;
    return (
      point.x >= crop.x &&
      point.x <= crop.x + crop.width &&
      point.y >= crop.y &&
      point.y <= crop.y + crop.height
    );
  }

  detectCropHandle(pointer, transform = this.getViewTransform()) {
    if (!this.state?.crop) {
      return null;
    }

    const crop = this.state.crop;
    const cropRect = {
      left: transform.left + crop.x * transform.scale,
      top: transform.top + crop.y * transform.scale,
      width: crop.width * transform.scale,
      height: crop.height * transform.scale,
    };
    const handles = createHandleMap(cropRect);

    for (const key of ["nw", "ne", "sw", "se"]) {
      const point = handles[key];
      if (Math.hypot(pointer.x - point.x, pointer.y - point.y) <= HANDLE_HIT_RADIUS) {
        return key;
      }
    }

    const nearTop = Math.abs(pointer.y - cropRect.top) <= HANDLE_HIT_RADIUS;
    const nearBottom = Math.abs(pointer.y - (cropRect.top + cropRect.height)) <= HANDLE_HIT_RADIUS;
    const nearLeft = Math.abs(pointer.x - cropRect.left) <= HANDLE_HIT_RADIUS;
    const nearRight = Math.abs(pointer.x - (cropRect.left + cropRect.width)) <= HANDLE_HIT_RADIUS;
    const withinX = pointer.x >= cropRect.left - HANDLE_HIT_RADIUS && pointer.x <= cropRect.left + cropRect.width + HANDLE_HIT_RADIUS;
    const withinY = pointer.y >= cropRect.top - HANDLE_HIT_RADIUS && pointer.y <= cropRect.top + cropRect.height + HANDLE_HIT_RADIUS;

    if (nearTop && Math.abs(pointer.x - handles.n.x) <= HANDLE_HIT_RADIUS * 1.5) {
      return "n";
    }
    if (nearBottom && Math.abs(pointer.x - handles.s.x) <= HANDLE_HIT_RADIUS * 1.5) {
      return "s";
    }
    if (nearLeft && Math.abs(pointer.y - handles.w.y) <= HANDLE_HIT_RADIUS * 1.5) {
      return "w";
    }
    if (nearRight && Math.abs(pointer.y - handles.e.y) <= HANDLE_HIT_RADIUS * 1.5) {
      return "e";
    }
    if (nearTop && withinX) {
      return "n";
    }
    if (nearBottom && withinX) {
      return "s";
    }
    if (nearLeft && withinY) {
      return "w";
    }
    if (nearRight && withinY) {
      return "e";
    }

    return null;
  }

  updateCursor() {
    if (this.pointerMode === "pan" || this.spacePressed) {
      this.canvas.style.cursor = "grabbing";
      return;
    }
    if (this.tool === "crop") {
      if (this.hoverHandle) {
        this.canvas.style.cursor = cursorForHandle(this.hoverHandle);
        return;
      }
      if (this.hoverImagePoint && this.isPointInsideCrop(this.hoverImagePoint)) {
        this.canvas.style.cursor = "grab";
        return;
      }
      this.canvas.style.cursor = "crosshair";
      return;
    }
    this.canvas.style.cursor = "crosshair";
  }

  updateHoverFromPointer(pointer) {
    if (!this.sourceImage) {
      this.hoverPointer = null;
      this.hoverImagePoint = null;
      this.hoverHandle = null;
      this.updateCursor();
      return;
    }

    const transform = this.getViewTransform();
    const imagePoint = this.screenToImage(pointer, transform);
    this.hoverPointer = pointer;
    this.hoverImagePoint = this.isPointInsideImage(imagePoint) ? imagePoint : null;
    this.hoverHandle = this.tool === "crop" ? this.detectCropHandle(pointer, transform) : null;
    this.updateCursor();
  }

  onPointerDown(event) {
    if (!this.sourceImage) {
      return;
    }

    this.ensureState();
    this.layoutStage();
    const pointer = this.getPointerPoint(event);
    this.updateHoverFromPointer(pointer);
    const imagePoint = this.hoverImagePoint ?? this.screenToImage(pointer);
    const crop = this.state.crop;

    if (event.button === 1 || this.spacePressed) {
      this.pointerMode = "pan";
      this.dragStart = {
        pointer,
        viewport: { ...this.state.viewport },
      };
      this.canvas.setPointerCapture(event.pointerId);
      this.updateCursor();
      return;
    }

    if (this.tool === "crop") {
      if (this.hoverHandle) {
        this.pointerMode = "resize";
        this.dragHandle = this.hoverHandle;
        this.dragStart = { crop: { ...crop } };
      } else if (imagePoint && this.isPointInsideCrop(imagePoint)) {
        this.pointerMode = "move";
        this.dragStart = {
          point: imagePoint,
          crop: { ...crop },
        };
      } else {
        return;
      }
      this.canvas.setPointerCapture(event.pointerId);
      this.updateCursor();
      return;
    }

    if (!imagePoint || !this.isPointInsideCrop(imagePoint)) {
      return;
    }

    this.pointerMode = "paint";
    this.activeStroke = {
      mode: this.tool === "erase" ? "erase" : "paint",
      radius_px: Number(this.brushSizeInput.value),
      softness: Number(this.softnessInput.value),
      opacity: DEFAULT_BRUSH_OPACITY,
      points: [[imagePoint.x, imagePoint.y]],
    };
    this.state.strokes.push(this.activeStroke);
    this.canvas.setPointerCapture(event.pointerId);
    this.scheduleSave();
    this.draw();
  }

  onPointerMove(event) {
    if (!this.sourceImage) {
      return;
    }

    this.layoutStage();
    const pointer = this.getPointerPoint(event);
    this.updateHoverFromPointer(pointer);
    const imagePoint = this.hoverImagePoint ?? this.screenToImage(pointer);

    if (!this.pointerMode) {
      this.draw();
      return;
    }

    if (this.pointerMode === "pan") {
      const transform = this.getViewTransform();
      this.state.viewport.pan_x = this.dragStart.viewport.pan_x + (pointer.x - this.dragStart.pointer.x) / transform.scale;
      this.state.viewport.pan_y = this.dragStart.viewport.pan_y + (pointer.y - this.dragStart.pointer.y) / transform.scale;
      this.scheduleSave();
      this.draw();
      return;
    }

    if (this.pointerMode === "move") {
      const dx = imagePoint.x - this.dragStart.point.x;
      const dy = imagePoint.y - this.dragStart.point.y;
      const crop = { ...this.dragStart.crop };
      crop.x = clamp(Math.round(crop.x + dx), 0, this.sourceImage.width - crop.width);
      crop.y = clamp(Math.round(crop.y + dy), 0, this.sourceImage.height - crop.height);
      this.state.crop = crop;
      this.scheduleSave();
      this.draw();
      return;
    }

    if (this.pointerMode === "resize") {
      this.state.crop = this.computeResizedCrop(imagePoint);
      this.scheduleSave();
      this.draw();
      return;
    }

    if (this.pointerMode === "paint" && this.activeStroke) {
      if (!imagePoint || !this.isPointInsideCrop(imagePoint)) {
        return;
      }
      const points = this.activeStroke.points;
      const minDistance = Math.max(1, this.activeStroke.radius_px * 0.12);
      const nextPoint = [imagePoint.x, imagePoint.y];
      if (normalizePointDistance(points[points.length - 1], nextPoint) >= minDistance) {
        points.push(nextPoint);
        this.scheduleSave();
        this.draw();
      }
    }
  }

  computeResizedCrop(imagePoint) {
    const ratio = parseRatio(this.getRatioValue());
    const source = this.dragStart.crop;
    const centerX = source.x + source.width / 2;
    const centerY = source.y + source.height / 2;
    let rawRect;

    const fitCorner = (anchor, point, handle) => {
      const widthAvailable = Math.abs(point.x - anchor.x);
      const heightAvailable = Math.abs(point.y - anchor.y);
      let width = widthAvailable;
      let height = Math.round((width * ratio.h) / ratio.w);
      if (height > heightAvailable) {
        height = heightAvailable;
        width = Math.round((height * ratio.w) / ratio.h);
      }
      let x = anchor.x;
      let y = anchor.y;
      if (handle.includes("w")) {
        x -= width;
      }
      if (handle.includes("n")) {
        y -= height;
      }
      return { x, y, width, height };
    };

    switch (this.dragHandle) {
      case "nw":
        rawRect = fitCorner({ x: source.x + source.width, y: source.y + source.height }, imagePoint, "nw");
        break;
      case "ne":
        rawRect = fitCorner({ x: source.x, y: source.y + source.height }, imagePoint, "ne");
        break;
      case "sw":
        rawRect = fitCorner({ x: source.x + source.width, y: source.y }, imagePoint, "sw");
        break;
      case "se":
        rawRect = fitCorner({ x: source.x, y: source.y }, imagePoint, "se");
        break;
      case "n": {
        const anchorY = source.y + source.height;
        const height = Math.max(1, Math.abs(anchorY - imagePoint.y));
        const width = Math.max(1, Math.round((height * ratio.w) / ratio.h));
        rawRect = { x: centerX - width / 2, y: anchorY - height, width, height };
        break;
      }
      case "s": {
        const anchorY = source.y;
        const height = Math.max(1, Math.abs(imagePoint.y - anchorY));
        const width = Math.max(1, Math.round((height * ratio.w) / ratio.h));
        rawRect = { x: centerX - width / 2, y: anchorY, width, height };
        break;
      }
      case "w": {
        const anchorX = source.x + source.width;
        const width = Math.max(1, Math.abs(anchorX - imagePoint.x));
        const height = Math.max(1, Math.round((width * ratio.h) / ratio.w));
        rawRect = { x: anchorX - width, y: centerY - height / 2, width, height };
        break;
      }
      case "e": {
        const anchorX = source.x;
        const width = Math.max(1, Math.abs(imagePoint.x - anchorX));
        const height = Math.max(1, Math.round((width * ratio.h) / ratio.w));
        rawRect = { x: anchorX, y: centerY - height / 2, width, height };
        break;
      }
      default:
        rawRect = source;
        break;
    }

    return fitCropNearCenter(rawRect, this.sourceImage.width, this.sourceImage.height, this.getRatioValue());
  }

  onPointerUp(event) {
    if (this.pointerMode) {
      this.canvas.releasePointerCapture?.(event.pointerId);
    }
    this.pointerMode = null;
    this.dragHandle = null;
    this.dragStart = null;
    this.activeStroke = null;
    this.updateCursor();
    this.scheduleSave();
    this.draw();
  }

  onPointerLeave() {
    if (this.pointerMode) {
      return;
    }
    this.hoverPointer = null;
    this.hoverImagePoint = null;
    this.hoverHandle = null;
    this.updateCursor();
    this.draw();
  }

  onWheel(event) {
    if (!this.sourceImage) {
      return;
    }
    this.ensureState();
    event.preventDefault();
    this.layoutStage();
    const pointer = this.getPointerPoint(event);
    const previousTransform = this.getViewTransform();
    const anchorImagePoint = this.screenToImage(pointer, previousTransform);
    const previous = this.state.viewport.zoom || 1;
    const next = clamp(previous * (event.deltaY < 0 ? 1.08 : 0.92), 0.2, 8);
    this.state.viewport = computeZoomAnchoredViewport({
      stageWidth: this.stageMetrics.cssWidth,
      stageHeight: this.stageMetrics.cssHeight,
      imageWidth: this.sourceImage.width,
      imageHeight: this.sourceImage.height,
      fitScale: previousTransform.fitScale,
      nextZoom: next,
      anchorCanvasPoint: pointer,
      anchorImagePoint,
    });
    this.scheduleSave();
    this.draw();
  }

  renderMaskOverlay(transform) {
    const crop = this.state.crop;
    const width = this.overlayCanvas.width;
    const height = this.overlayCanvas.height;
    if (!width || !height) {
      return;
    }

    const dpr = this.stageMetrics.dpr || 1;
    const cropLeft = Math.max(0, Math.floor((transform.left + crop.x * transform.scale) * dpr));
    const cropTop = Math.max(0, Math.floor((transform.top + crop.y * transform.scale) * dpr));
    const cropRight = Math.min(width, Math.ceil((transform.left + (crop.x + crop.width) * transform.scale) * dpr));
    const cropBottom = Math.min(height, Math.ceil((transform.top + (crop.y + crop.height) * transform.scale) * dpr));
    const cropClip = { left: cropLeft, top: cropTop, right: cropRight, bottom: cropBottom };
    const alphaBuffer = new Float32Array(width * height);

    const paintStroke = (stroke) => {
      const points = stroke.points || [];
      if (!points.length) {
        return;
      }

      const radius = Math.max(0.75, Number(stroke.radius_px || DEFAULT_BRUSH_RADIUS) * transform.scale * dpr);
      const softness = clamp(Number(stroke.softness ?? DEFAULT_BRUSH_SOFTNESS), 0, 1);
      const opacity = clamp(Number(stroke.opacity ?? DEFAULT_BRUSH_OPACITY), 0, 1);
      const mode = stroke.mode === "erase" ? "erase" : "paint";
      const toDevicePoint = (point) => {
        const screen = this.imageToScreen({ x: point[0], y: point[1] }, transform);
        return { x: screen.x * dpr, y: screen.y * dpr };
      };

      if (points.length === 1) {
        const point = toDevicePoint(points[0]);
        applyBrushDabToAlphaBuffer(alphaBuffer, width, point.x, point.y, radius, softness, opacity, mode, cropClip);
        return;
      }

      for (let index = 0; index < points.length - 1; index += 1) {
        const start = toDevicePoint(points[index]);
        const end = toDevicePoint(points[index + 1]);
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const distance = Math.hypot(dx, dy);
        const step = Math.max(1, radius * 0.25);
        const steps = Math.max(1, Math.ceil(distance / step));
        for (let stepIndex = 0; stepIndex <= steps; stepIndex += 1) {
          const t = stepIndex / steps;
          applyBrushDabToAlphaBuffer(
            alphaBuffer,
            width,
            start.x + dx * t,
            start.y + dy * t,
            radius,
            softness,
            opacity,
            mode,
            cropClip
          );
        }
      }
    };

    for (const stroke of this.state.strokes) {
      if (stroke.points?.length) {
        paintStroke(stroke);
      }
    }

    const imageData = this.overlayCtx.createImageData(width, height);
    const data = imageData.data;
    for (let index = 0; index < alphaBuffer.length; index += 1) {
      const alpha = clamp(alphaBuffer[index], 0, 1);
      if (alpha <= 0) {
        continue;
      }
      const offset = index * 4;
      data[offset] = PAINT_RGBA[0];
      data[offset + 1] = PAINT_RGBA[1];
      data[offset + 2] = PAINT_RGBA[2];
      data[offset + 3] = Math.round(alpha * PAINT_MAX_ALPHA * 255);
    }
    this.overlayCtx.clearRect(0, 0, width, height);
    this.overlayCtx.putImageData(imageData, 0, 0);
    this.ctx.drawImage(this.overlayCanvas, 0, 0, this.stageMetrics.cssWidth, this.stageMetrics.cssHeight);
  }

  drawCropOverlay(transform) {
    const crop = this.state.crop;
    const cropRect = {
      left: transform.left + crop.x * transform.scale,
      top: transform.top + crop.y * transform.scale,
      width: crop.width * transform.scale,
      height: crop.height * transform.scale,
    };

    this.ctx.save();
    this.ctx.fillStyle = "rgba(0, 0, 0, 0.52)";
    this.ctx.beginPath();
    this.ctx.rect(transform.left, transform.top, this.sourceImage.width * transform.scale, this.sourceImage.height * transform.scale);
    this.ctx.rect(cropRect.left, cropRect.top, cropRect.width, cropRect.height);
    this.ctx.fill("evenodd");

    this.ctx.strokeStyle = CROP_STROKE;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(cropRect.left, cropRect.top, cropRect.width, cropRect.height);

    this.ctx.strokeStyle = CROP_GRID;
    this.ctx.beginPath();
    this.ctx.moveTo(cropRect.left + cropRect.width / 3, cropRect.top);
    this.ctx.lineTo(cropRect.left + cropRect.width / 3, cropRect.top + cropRect.height);
    this.ctx.moveTo(cropRect.left + (cropRect.width * 2) / 3, cropRect.top);
    this.ctx.lineTo(cropRect.left + (cropRect.width * 2) / 3, cropRect.top + cropRect.height);
    this.ctx.moveTo(cropRect.left, cropRect.top + cropRect.height / 3);
    this.ctx.lineTo(cropRect.left + cropRect.width, cropRect.top + cropRect.height / 3);
    this.ctx.moveTo(cropRect.left, cropRect.top + (cropRect.height * 2) / 3);
    this.ctx.lineTo(cropRect.left + cropRect.width, cropRect.top + (cropRect.height * 2) / 3);
    this.ctx.stroke();

    const handles = createHandleMap(cropRect);
    Object.entries(handles).forEach(([handle, point]) => {
      this.ctx.save();
      const active = this.hoverHandle === handle || this.dragHandle === handle;
      this.ctx.fillStyle = active ? "#9ef5ff" : "#dcbf7a";
      this.ctx.strokeStyle = active ? "#e7fdff" : "#614c1b";
      this.ctx.lineWidth = 1.5;
      if (["n", "e", "s", "w"].includes(handle)) {
        this.ctx.beginPath();
        this.ctx.rect(point.x - 5, point.y - 5, 10, 10);
        this.ctx.fill();
        this.ctx.stroke();
      } else {
        this.ctx.beginPath();
        this.ctx.arc(point.x, point.y, HANDLE_DRAW_RADIUS, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();
      }
      this.ctx.restore();
    });

    this.ctx.restore();
  }

  drawBrushPreview(transform) {
    if (!this.hoverImagePoint || !["paint", "erase"].includes(this.tool)) {
      return;
    }
    if (!this.isPointInsideCrop(this.hoverImagePoint)) {
      return;
    }

    const screen = this.imageToScreen(this.hoverImagePoint, transform);
    const radius = Number(this.brushSizeInput.value) * transform.scale;
    const softness = Number(this.softnessInput.value);
    const innerRadius = radius * Math.max(0.2, 1 - softness);

    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    this.ctx.strokeStyle = this.tool === "erase" ? ERASE_STROKE : PAINT_STROKE;
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash(this.tool === "erase" ? [7, 5] : []);
    this.ctx.stroke();

    if (this.tool === "paint") {
      this.ctx.beginPath();
      this.ctx.arc(screen.x, screen.y, innerRadius, 0, Math.PI * 2);
      this.ctx.strokeStyle = "rgba(160, 245, 255, 0.55)";
      this.ctx.lineWidth = 1;
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.arc(screen.x, screen.y, 2.5, 0, Math.PI * 2);
      this.ctx.fillStyle = PAINT_STROKE;
      this.ctx.fill();
    }

    this.ctx.restore();
  }

  draw() {
    this.layoutStage();
    this.ctx.clearRect(0, 0, this.stageMetrics.cssWidth, this.stageMetrics.cssHeight);

    if (!this.sourceImage) {
      this.cropInfo.textContent = "No image selected";
      this.ctx.fillStyle = "rgba(255, 255, 255, 0.12)";
      this.ctx.font = "16px Georgia";
      this.ctx.fillText("Select an input image to begin.", 24, 36);
      return;
    }

    this.ensureState();
    const transform = this.getViewTransform();
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.rect(0, 0, this.stageMetrics.cssWidth, this.stageMetrics.cssHeight);
    this.ctx.clip();
    this.ctx.drawImage(
      this.sourceImage,
      transform.left,
      transform.top,
      this.sourceImage.width * transform.scale,
      this.sourceImage.height * transform.scale
    );
    this.renderMaskOverlay(transform);
    this.drawCropOverlay(transform);
    this.drawBrushPreview(transform);
    this.ctx.restore();

    const crop = this.state.crop;
    this.cropInfo.textContent = `${crop.width}x${crop.height} px | ${this.getRatioValue()} | ${this.tool} | brush ${Number(this.brushSizeInput.value)} px | ${this.state.strokes.length} stroke(s)`;
  }
}

app.registerExtension({
  name: "Burve.CropMaskLoad",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TARGET_NODE) {
      return;
    }

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function (...args) {
      this.__burveCropMaskEditor?.destroy?.();
      this.__burveCropMaskEditor = null;
      return onRemoved?.apply(this, args);
    };
  },
  nodeCreated(node) {
    if (node.comfyClass !== TARGET_NODE || node.__burveCropMaskEditor) {
      return;
    }
    node.__burveCropMaskEditor = new CropMaskEditor(node);
  },
});
