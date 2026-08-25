# July 24, 2026 — Gemini Image 3.0

This release modernizes both Google image nodes around Google's stable image model IDs, google-genai 2.13, and the interactive image APIs while preserving existing workflows.

## Model endpoint repair

- New-node default: `gemini-3.1-flash-image`.
- Stable catalog: `gemini-3.1-flash-image`, `gemini-3-pro-image`, `gemini-3.1-flash-lite-image`, and legacy `gemini-2.5-flash-image`.
- Saved `gemini-3.1-flash-image-preview` and `gemini-3-pro-image-preview` selections are normalized to their stable IDs before validation and SDK calls.
- Gemini 2.5 remains selectable and reports its October 2, 2026 Vertex retirement. It is never silently replaced.
- Stable Gemini 3.x image calls on Vertex use `global` automatically. Strict configured-location mode rejects a non-global 3.x call before contacting Google.
- Lazy model discovery is cached per provider/location/model. A model-lookup outage does not block generation; endpoint 404 diagnostics include requested and normalized IDs, provider, API mode, and effective location.

## API and storage behavior

- Dependency range is now `google-genai>=2.13.0,<3`; startup validates required SDK symbols and the Interactions client.
- AI Studio `auto` routing uses Interactions for Gemini 3.1 Flash Image and Gemini 3 Pro Image. Flash Lite and Gemini 2.5 use GenerateContent.
- Vertex `auto` routing uses GenerateContent. Explicit Vertex Interactions remains experimental and is limited to supported Flash/Pro models.
- Stateless Interactions explicitly send `store=false`. Stateful mode is opt-in, sends `store=true`, and produces a resumable provider/model/API-bound interaction pipe.
- A non-stored interaction pipe cannot be resumed. Provider, model, and API mismatches fail locally.
- Interactions are not automatically retried after an ambiguous transport result. GenerateContent keeps bounded retry behavior and reports attempt, API mode, and retry reason.

## Model-aware features

- Gemini 3.1 Flash Image: `512`, `1K`, `2K`, `4K`; extended `1:4`, `1:8`, `4:1`, `8:1` ratios; web/image search; thinking controls; native video or public YouTube input; up to 10 object and 4 character references.
- Gemini 3 Pro Image: `1K`, `2K`, `4K`; web search; implicit thinking; ordered interleaved output; up to 6 object, 5 character, and 3 style references.
- Gemini 3.1 Flash Lite Image: `1K`; the standard ten ratios; thinking controls; up to 14 object/generic references; no search or video.
- Gemini 2.5 Flash Image: `1K`; the standard ten ratios; up to 3 references; no search, thinking, video, or Interactions.
- The non-primary `9:21` spelling is not exposed.
- Added image-only response mode plus bounded temperature and top-p controls.
- Added validated PDF/TXT/Markdown/CSV context files.
- Added Vertex PNG, JPEG, WebP, HEIC, and HEIF preservation/decoding where the model permits them, JPEG quality, person-generation policy, and prominent-people policy.

## Nodes and outputs

The existing `BurveGoogleImageGen` and `BurveVertexImageGen` node IDs, old input names, and first five output positions are unchanged. New optional sockets are appended.

New outputs:

- `response_text`
- `grounding_metadata_json`
- `interleaved_content`
- `interaction_pipe`

New helper nodes:

- **Burve Gemini Reference Pack**
- **Burve Gemini Input Files**
- **Burve List Google Image Models**

Reference precedence is Character Generation Pipe, then Gemini Reference Pack, then legacy IMAGE_LIST. A conflict is reported instead of silently combining sources.

## Response and attribution changes

- `GENERATED_IMAGE_PIPE` version 2 preserves original bytes and MIME type for both providers, ordered response blocks, thought/final designation, dimensions, grounding, API mode, interaction ID, and effective location.
- Version 1 pipes remain accepted by the save node.
- Identical thought and final images are preserved as distinct semantic outputs.
- Differently sized results remain individually available in the content and generated-image pipes; only compatible dimensions are batched on the legacy IMAGE socket.
- Grounding output includes queries, image-search steps, sources, citations, usage, and Google's `search_suggestions` HTML.
- Both Gemini nodes render attribution in a sandboxed frame with scripts disabled. Image Search is rejected when this renderer is unavailable.
- Previous grounding UI is cleared at execution start and whenever the next response contains no grounding.

## Compatibility and dependencies

- Minimum ComfyUI is `0.4.0`, the first tagged release containing the native V3 VIDEO/VideoInput interface used here.
- Added `pillow-heif>=0.18.0,<2`.
- API keys, prompts, uploaded media, and interaction contents are not written to plugin logs.
- Batch API, background Interactions, generic function calling, structured JSON generation, Imagen, Veo, and audio generation remain outside this release.

## Validation

- Offline unit and frontend tests cover catalogs, aliases, lifecycle/location rules, request contracts, storage/continuation, feature limits, response ordering, malformed parts, mixed dimensions, MIME preservation, saved workflow fixtures, attribution rendering, and stale-state clearing.
- The request builders are also validated against the official google-genai 2.13 wheel without sending paid calls.
- `test_gemini_live.py` contains opt-in paid smoke tests. They remain skipped unless `BURVE_RUN_GEMINI_LIVE_TESTS=1` is set and report unavailable/account-restricted models as skips.

Documentation basis: [image generation](https://ai.google.dev/gemini-api/docs/image-generation), [model deprecations](https://ai.google.dev/gemini-api/docs/deprecations), [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview), and [google-genai 2.13](https://pypi.org/project/google-genai/).
