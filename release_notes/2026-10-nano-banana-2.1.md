# 3.1.0 — Nano Banana 2.1

Both Google image nodes now include `gemini-nano-banana-2.1`, and newly created nodes select it by default. Existing selections remain on their saved model. This update follows Google's documentation checked on October 13, 2026.

## Model controls and inputs

- Resolutions: `1K` (default), `2K`, and `4K`; `512` is rejected.
- Thinking: `MINIMAL`, `MEDIUM` (default), or `HIGH`, with the existing `include_thoughts` toggle.
- Search: `off`, `web`, `image`, or `web+image`. Image Search retains the bundled attribution-renderer requirement.
- Ratios: the standard ten plus `1:4`, `1:8`, `4:1`, and `8:1`. Vertex also accepts `9:21`; AI Studio rejects that ratio locally.
- References: up to 14 total, including at most 10 object and 4 character references. Style-labelled references remain unsupported for this model.
- Existing native video, public YouTube URL, and PDF/TXT/Markdown/CSV document inputs work with 2.1. Connecting both video inputs remains invalid.
- The DynamicCombo exposes `aspect_ratio`, `resolution`, `search_mode`, `thinking_level`, `include_thoughts`, and `output_mime_type`. It adds no nested prominent-people control.

## Provider behavior

- AI Studio `auto` uses Interactions; explicit GenerateContent is also supported. Stateful edits use the existing stored interaction pipe and provider/model/API validation. Stateless Interactions continue to send `store=false`.
- Vertex 2.1 uses GenerateContent only and requires `global`. `auto` selects that location; strict configured mode rejects a non-global location. Interactions and stateful continuation fail locally for this model.
- Vertex 2.1 requests omit `seed`, `temperature`, and `top_p`, which Google documents as unsupported. The existing widgets remain available for saved-workflow compatibility, and `system_messages` explains that their values are ignored. AI Studio sampling behavior is unchanged.
- The nested output format is `default` or `image/jpeg`. AI Studio Interactions accepts explicit JPEG; AI Studio GenerateContent ignores the override with a diagnostic. Vertex accepts `default`, PNG, or JPEG, with top-level Vertex controls taking precedence and JPEG quality applied only to JPEG. WebP, HEIC, and HEIF output requests are rejected for Vertex 2.1; input MIME support does not imply selectable output support.
- Existing Vertex person-generation and prominent-people policies remain available. `ALLOW_NONE` takes precedence over allowing prominent people.

## Compatibility and lifecycle

- `BurveGoogleImageGen` and `BurveVertexImageGen` retain their node IDs, original input order, and all nine output positions. No sockets or dependencies are added.
- Existing stable and legacy models remain selectable. Retired Flash/Pro preview IDs still resolve to their corresponding stable IDs and retain their compatibility routing; they are not redirected to 2.1.
- `gemini-3.1-flash-image` is marked deprecated, with no announced shutdown date and 2.1 as its replacement. The `gemini-2.5-flash-image` Vertex retirement date is corrected to March 15, 2027.
- Model discovery recognizes the new model ID even though it does not contain `image`, and reports provider-specific capabilities.
- Earlier release notes remain historical records. Batch jobs, new video/file workflows, and changes to older-model API behavior are outside this update.

## Documentation basis

- [Nano Banana 2.1 model](https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1)
- [Image generation, thinking, and reference limits](https://ai.google.dev/gemini-api/docs/image-generation)
- [GenerateContent image generation](https://ai.google.dev/gemini-api/docs/generate-content/image-generation)
- [Interactions response-format schema](https://ai.google.dev/api/interactions-api#ImageResponseFormat)
- [Nano Banana 2.1 on Google Cloud: sampling, ratios, and location](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/nano-banana-2-1)
- [Cloud image output and person-generation controls](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/Shared.Types/GeminiExample#ImageConfig)
- [Gemini API October 6 release notes](https://ai.google.dev/gemini-api/docs/changelog#october-6-2026)
- [Gemini 2.5 Flash Image lifecycle on Google Cloud](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash-image)
