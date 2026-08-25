export const GEMINI_GROUNDING_RENDERER_VERSION = "burve-grounding-v1";

export function normalizeGroundingMessage(message) {
  const raw = message?.burve_gemini_grounding ?? message;
  const payload = Array.isArray(raw) ? raw[0] : raw;
  if (!payload || payload.clear) {
    return {
      clear: true,
      rendererVersion: GEMINI_GROUNDING_RENDERER_VERSION,
      suggestions: [],
      citations: [],
    };
  }

  const metadata = payload.metadata && typeof payload.metadata === "object"
    ? payload.metadata
    : {};
  const suggestions = Array.isArray(metadata.search_suggestions_html)
    ? metadata.search_suggestions_html.filter((value) => typeof value === "string" && value.trim())
    : [];
  const citations = normalizeCitations(metadata);

  return {
    clear: suggestions.length === 0 && citations.length === 0,
    rendererVersion: payload.renderer_version || GEMINI_GROUNDING_RENDERER_VERSION,
    suggestions,
    citations,
  };
}

export function buildAttributionDocument(payload) {
  const suggestions = (payload?.suggestions || []).join("\n");
  const citations = (payload?.citations || [])
    .filter((citation) => citation && (citation.url || citation.uri))
    .map((citation) => {
      const url = escapeHtml(citation.url || citation.uri);
      const title = escapeHtml(citation.title || citation.source || url);
      return `<li><a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a></li>`;
    })
    .join("");

  return `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<base target="_blank">
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 10px; background: #181a1f; color: #e8e8e8; font: 12px/1.45 system-ui, sans-serif; }
  h4 { margin: 0 0 8px; color: #c9d5ff; font-size: 12px; }
  ul { margin: 8px 0 0; padding-left: 20px; }
  a { color: #91b7ff; overflow-wrap: anywhere; }
</style>
</head>
<body>
  <h4>Google Search grounding</h4>
  <div data-google-search-suggestions>${suggestions}</div>
  ${citations ? `<ul>${citations}</ul>` : ""}
</body>
</html>`;
}

function extractGenerateContentCitations(grounding) {
  const chunks = Array.isArray(grounding?.grounding_chunks)
    ? grounding.grounding_chunks
    : [];
  return chunks
    .map((chunk) => chunk?.web || chunk?.retrieved_context || null)
    .filter(Boolean)
    .map((source) => ({
      title: source.title || "",
      url: source.uri || source.url || "",
    }));
}

function normalizeCitations(metadata) {
  const citations = Array.isArray(metadata?.citations) ? metadata.citations : [];
  const direct = citations.flatMap((citation) => {
    if (!citation || typeof citation !== "object") {
      return [];
    }
    if (citation.url || citation.uri) {
      return [citation];
    }
    if (Array.isArray(citation.sources)) {
      return citation.sources;
    }
    if (citation.source && typeof citation.source === "object") {
      return [citation.source];
    }
    return [];
  });
  if (direct.length > 0) {
    return direct;
  }
  if (Array.isArray(metadata?.sources) && metadata.sources.length > 0) {
    return metadata.sources;
  }
  return extractGenerateContentCitations(metadata?.grounding);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
