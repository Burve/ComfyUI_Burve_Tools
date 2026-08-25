import { app } from "../../../scripts/app.js";
import {
  buildAttributionDocument,
  GEMINI_GROUNDING_RENDERER_VERSION,
  normalizeGroundingMessage,
} from "./burve_gemini_grounding_helpers.js";

const TARGET_NODES = new Set(["BurveGoogleImageGen", "BurveVertexImageGen"]);
const MARKER_WIDGET = "grounding_renderer";
const DOM_WIDGET = "gemini_grounding_attribution";

function hideMarkerWidget(node) {
  const widget = node.widgets?.find((item) => item.name === MARKER_WIDGET);
  if (!widget) {
    return;
  }
  widget.value = GEMINI_GROUNDING_RENDERER_VERSION;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
}

function ensureAttributionWidget(node) {
  if (node.__burveGeminiGroundingWidget) {
    return node.__burveGeminiGroundingWidget;
  }
  if (typeof node.addDOMWidget !== "function") {
    return null;
  }

  const iframe = document.createElement("iframe");
  iframe.title = "Google Search grounding";
  iframe.setAttribute("sandbox", "allow-popups allow-popups-to-escape-sandbox");
  iframe.referrerPolicy = "no-referrer";
  iframe.style.cssText = [
    "width:100%",
    "height:180px",
    "border:1px solid #383c46",
    "border-radius:6px",
    "background:#181a1f",
    "display:none",
  ].join(";");

  const widget = node.addDOMWidget(DOM_WIDGET, "iframe", iframe, {
    serialize: false,
    hideOnZoom: false,
  });
  widget.computeSize = () => [node.size?.[0] || 420, iframe.style.display === "none" ? 0 : 190];
  node.__burveGeminiGroundingWidget = { iframe, widget };
  return node.__burveGeminiGroundingWidget;
}

function clearAttribution(node) {
  const groundingWidget = ensureAttributionWidget(node);
  if (!groundingWidget) {
    return;
  }
  groundingWidget.iframe.removeAttribute("srcdoc");
  groundingWidget.iframe.style.display = "none";
  node.setDirtyCanvas?.(true, true);
}

function renderAttribution(node, message) {
  const payload = normalizeGroundingMessage(message);
  if (payload.clear) {
    clearAttribution(node);
    return;
  }
  const groundingWidget = ensureAttributionWidget(node);
  if (!groundingWidget) {
    return;
  }
  groundingWidget.iframe.srcdoc = buildAttributionDocument(payload);
  groundingWidget.iframe.style.display = "block";
  node.setSize?.([Math.max(node.size?.[0] || 420, 420), node.computeSize?.()[1] || node.size?.[1]]);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "burve.gemini.grounding",
  nodeCreated(node) {
    if (!TARGET_NODES.has(node.comfyClass)) {
      return;
    }

    hideMarkerWidget(node);
    ensureAttributionWidget(node);

    const previousExecutionStart = node.onExecutionStart;
    node.onExecutionStart = function onExecutionStart(...args) {
      clearAttribution(this);
      return previousExecutionStart?.apply(this, args);
    };

    const previousExecuted = node.onExecuted;
    node.onExecuted = function onExecuted(message, ...args) {
      const result = previousExecuted?.call(this, message, ...args);
      renderAttribution(this, message);
      return result;
    };
  },
});
