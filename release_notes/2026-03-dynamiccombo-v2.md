# March 2026 DynamicCombo V2 Release

## Release Summary

- Migrated the package from V1 `NODE_CLASS_MAPPINGS` loading to a V3 `comfy_entrypoint()` extension.
- Rebuilt `Burve Google Image Gen` and `Burve Google Image Gen (Vertex AI)` around `IO.DynamicCombo.Input(...)` so each Gemini model now exposes only its documented model-specific controls.
- Removed the old legacy Gemini widgets and compatibility shims for static `resolution`, `aspect_ratio`, search toggles, and thinking toggles.
- Added a shared Gemini image service for model specs, request building, and response parsing across AI Studio and Vertex.
- Pinned `google-genai` to `>=1.68.0,<2` for documented thinking config, `SearchTypes`, `ImageSearch`, and `ProminentPeople`.
- Standardized fallback text outputs:
  - `thinking_process`: `No thinking output returned by the model.`
  - `system_messages`: `No system messages.`

## Breaking Changes

- The package now requires a DynamicCombo-capable ComfyUI V3 build.
- Saved workflows for the two Gemini image nodes will not keep removed legacy widget values from the old static schema.
- `0.5K` was replaced with the documented `512` value for `gemini-3.1-flash-image-preview`.

## Model-Specific UI

- `gemini-2.5-flash-image`: `aspect_ratio`
- `gemini-3-pro-image-preview`: `aspect_ratio`, `resolution`, `search_mode`
- `gemini-3.1-flash-image-preview`: `aspect_ratio`, `resolution`, `search_mode`, `thinking_level`, `include_thoughts`, `output_mime_type`, `prominent_people`

## Patch Note

- Corrected `gemini-3-pro-image-preview` on both Gemini image nodes to stop exposing and sending thinking controls. The model may still return thought text or thought images in the existing outputs, but the request payload no longer includes `ThinkingConfig` for this model.
- Updated the Gemini image nodes to use standard ComfyUI-compatible `control_after_generate=True` seed metadata, fixing the seed control regression without a custom frontend workaround.
