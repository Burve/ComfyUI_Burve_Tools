# ComfyUI Burve Tools

A collection of custom nodes for ComfyUI, focusing on integration with Google's Gemini AI models for image generation and more.

## Nodes

### Gemini Image — Nano Banana 2.1 update

Both Google image-generation nodes now support **Nano Banana 2.1** (`gemini-nano-banana-2.1`), the default for newly created nodes. Existing model selections and saved workflows remain compatible. The catalog below describes the plugin's supported features, with the Nano Banana 2.1 update verified against Google's public documentation on **October 13, 2026**. See the [release notes](release_notes/2026-10-nano-banana-2.1.md) for the provider-specific details.

| Model ID | Lifecycle | Output sizes | Search / thinking / video | Reference limits | Vertex location |
| --- | --- | --- | --- | --- | --- |
| `gemini-nano-banana-2.1` | Stable; new-node default | `1K`, `2K`, `4K` | Web + image search; `MINIMAL`/`MEDIUM`/`HIGH` thinking, default `MEDIUM`; video-to-image | 14 total; up to 10 object and 4 character | `global` |
| `gemini-3.1-flash-image` | Deprecated; no shutdown date announced | `512`, `1K`, `2K`, `4K` | Web + image search; `MINIMAL`/`HIGH` thinking; video-to-image | 14 total; up to 10 object and 4 character | `global` |
| `gemini-3-pro-image` | Stable | `1K`, `2K`, `4K` | Web search; implicit model-managed thinking; interleaved text/image | 14 total; up to 6 object, 5 character, and 3 style | `global` |
| `gemini-3.1-flash-lite-image` | Stable | `1K` | `MINIMAL`/`HIGH` thinking; no search or video | 14 object/generic | `global` |
| `gemini-2.5-flash-image` | Legacy; Vertex retirement **March 15, 2027** | `1K` | No search, thinking, video, or Interactions | 3 total | Configured region |

Saved `gemini-3.1-flash-image-preview` and `gemini-3-pro-image-preview` selections are normalized to their stable IDs before validation and SDK calls. The migration is reported in `system_messages`; existing selections are never silently upgraded to 2.1. The standard ratio set is `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, and `21:9`; Nano Banana 2.1 and Gemini 3.1 Flash Image additionally support `1:4`, `1:8`, `4:1`, and `8:1`. Nano Banana 2.1 also supports `9:21` on Vertex; AI Studio rejects that ratio before sending a request.

### 1. Burve Google Image Gen

The AI Studio node supports GenerateContent and the Interactions API. Its existing node ID, original input order, and first five outputs remain compatible with saved workflows.

#### Shared Gemini inputs

These inputs are present on both Google image nodes:

| Input | Type / values | Purpose and behavior |
| --- | --- | --- |
| `prompt` | `STRING` | Image-generation or editing request. Ignored when `character_pipe` is connected. |
| `model` | DynamicCombo | Selects the model and exposes only that model's supported fields. |
| `seed` | `INT` | Generation seed with ComfyUI's after-generate control. Nano Banana 2.1 on Vertex omits it from the API request and reports that it is ignored. |
| `system_instructions` | `STRING`, optional | Higher-level model instructions. Ignored when `character_pipe` is connected. |
| `reference_images` | `IMAGE_LIST`, optional | Legacy unlabelled image references, normally created by **Burve Image Reference Pack**. |
| `character_pipe` | `CHARACTER_GEN_PIPE`, optional | One-cable prompt, instructions, references, and planner metadata from **Burve Character Planner**. It overrides direct prompt/instruction/reference inputs. |
| `aspect_ratio_override` | `STRING`, optional | Overrides the model's nested `aspect_ratio`; intended for **Burve Crop + Mask Load** or **Burve Image Info**. Unsupported ratios fail before the API call. |
| `request_timeout_seconds` | `INT`, `10..1800`, default `120` | Overall request budget, including GenerateContent retries. |
| `retry_attempts` | `INT`, `1..10`, default `5` | Total GenerateContent attempts for transient failures; `1` disables retries. Interactions are never automatically retried. |
| `api_mode` | `auto`, `generate_content`, `interactions` | Chooses the Google API surface. `auto` is recommended. |
| `response_mode` | `text_and_image`, `image_only` | Requests text plus image or image-only output. |
| `conversation_mode` | `stateless`, `stateful` | Controls Interactions storage. Stateful mode requires the Interactions API. |
| `previous_interaction` | `GEMINI_INTERACTION_PIPE`, optional | Resumes a stored Interaction from the same provider, model, and API mode. |
| `reference_pack` | `GEMINI_REFERENCE_PIPE`, optional | Role-labelled `generic`, `object`, `character`, or `style` references. |
| `input_files` | `GEMINI_INPUT_FILES_PIPE`, optional | Validated PDF, TXT, Markdown, or CSV context. Documents are unsupported by Gemini 2.5 Image. |
| `video` | `VIDEO`, optional | Native ComfyUI video context; Nano Banana 2.1 or Gemini 3.1 Flash Image. |
| `video_url` | `STRING`, optional | Public HTTPS YouTube URL; Nano Banana 2.1 or Gemini 3.1 Flash Image. Connecting both video inputs is rejected. |
| `temperature` | `FLOAT`, `0.0..2.0`, default `1.0` | Controls sampling variability; omitted with a diagnostic for Nano Banana 2.1 on Vertex. |
| `top_p` | `FLOAT`, `0.0..1.0`, default `0.95` | Controls nucleus sampling; omitted with a diagnostic for Nano Banana 2.1 on Vertex. |
| `grounding_renderer` | `STRING`, advanced/internal | Frontend-managed Image Search attribution version. Do not connect or edit manually. |

Reference precedence is **Character Generation Pipe → Gemini Reference Pack → legacy IMAGE_LIST**. If more than one source is connected, the winning source is used and a warning is added to `system_messages`.

#### Model-specific DynamicCombo fields

| Model | Nested fields |
| --- | --- |
| `gemini-nano-banana-2.1` | `aspect_ratio`, `resolution`, `search_mode`, `thinking_level`, `include_thoughts`, `output_mime_type` |
| `gemini-3.1-flash-image` | `aspect_ratio`, `resolution`, `search_mode`, `thinking_level`, `include_thoughts`, `output_mime_type`, `prominent_people` |
| `gemini-3-pro-image` | `aspect_ratio`, `resolution`, `search_mode`, `output_mime_type` |
| `gemini-3.1-flash-lite-image` | `aspect_ratio`, fixed `resolution=1K`, `thinking_level`, `include_thoughts` |
| `gemini-2.5-flash-image` | `aspect_ratio` |

`search_mode` supports `off`/`web` on Pro and `off`/`web`/`image`/`web+image` on Nano Banana 2.1 and 3.1 Flash. Image Search is accepted only when the bundled attribution renderer is active. Nano Banana 2.1 has `MINIMAL`, `MEDIUM`, and `HIGH` thinking, defaulting to `MEDIUM`; 3.1 Flash and Flash Lite retain `MINIMAL`/`HIGH`. `include_thoughts` requests available thought summaries/content.

Nano Banana 2.1 defaults to `resolution=1K` and does not accept `512`. Its nested `output_mime_type` offers `default` or `image/jpeg`; explicit JPEG is supported by AI Studio Interactions, while AI Studio GenerateContent ignores this nested format override with a diagnostic. On Vertex, the top-level format control takes precedence. The older models retain their existing nested MIME and prominent-people fields for workflow compatibility; 2.1 adds no nested prominent-people field.

#### API routing and storage

| Provider | Nano Banana 2.1 in `auto` | Flash 3.1 / Pro in `auto` | Flash Lite / 2.5 in `auto` | Explicit Interactions |
| --- | --- | --- | --- | --- |
| AI Studio | Interactions | Interactions | GenerateContent | Nano Banana 2.1, Flash 3.1, and Pro |
| Vertex AI | GenerateContent | GenerateContent | GenerateContent | Experimental; Flash 3.1 and Pro only |

Nano Banana 2.1 supports explicit GenerateContent on both providers. Vertex 2.1 rejects Interactions and stateful continuation locally; AI Studio 2.1 supports stored Interactions and continuation with the existing interaction pipe.

Stateless Interactions explicitly send `store=false`. Selecting `stateful` sends `store=true` and creates a resumable pipe; only a stored pipe can be connected to `previous_interaction`. GenerateContent retains bounded retries and reports the attempt/API/retry reason. Interactions use one attempt because an ambiguous transport failure may already have created the stored request.

Google-supplied search suggestions and citation links are shown in a sandboxed, script-disabled attribution frame and returned as JSON. Missing attribution support causes Image Search to fail locally before a non-compliant request is sent.

### 2. Burve Google Image Gen (Vertex AI)

The Vertex node has all shared inputs above plus these Vertex-specific controls:

| Input | Values | Purpose and behavior |
| --- | --- | --- |
| `vertex_location_mode` | `auto`, `configured` | `auto` routes Nano Banana 2.1 and Gemini 3.x image models to `global`; strict configured mode rejects a non-global call for these models. |
| `vertex_output_mime_type` | `default`, PNG, JPEG, WebP, HEIC, HEIF | Selects the response format where the chosen model supports it. Nano Banana 2.1 accepts only `default`, PNG, or JPEG; Gemini 2.5 does not support HEIC/HEIF. Unsupported choices fail locally. |
| `output_compression_quality` | `INT`, `0..100`, default `90` | JPEG quality for Vertex GenerateContent; ignored for non-JPEG formats and Interactions. |
| `person_generation` | `default`, `ALLOW_ALL`, `ALLOW_ADULT`, `ALLOW_NONE` | Vertex GenerateContent person-generation policy. |
| `prominent_people` | `default`, `allow`, `block` | Vertex GenerateContent prominent/public-person policy. `ALLOW_NONE` person generation takes precedence. |

The node authenticates with `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and Google credentials / ADC. It never reads `GEMINI_API_KEY`. Explicit Vertex Interactions remains experimental; `auto` uses GenerateContent.

Nano Banana 2.1 on Vertex does not support sampling parameters. The plugin retains the shared `seed`, `temperature`, and `top_p` widgets for workflow compatibility, omits those values from Vertex 2.1 requests, and explains this in `system_messages`. AI Studio sampling behavior and all older-model behavior remain unchanged. Vertex 2.1 supports the existing person-generation and prominent-people controls and JPEG compression quality. The broader image-input MIME support is separate from selectable output formats.

#### Gemini outputs and pipes

Both Google image nodes expose the same nine outputs. The original first five remain in place and four newer outputs are appended.

| Output | Type | Contents and recommended use |
| --- | --- | --- |
| `image` | `IMAGE` | Final images with compatible dimensions, batched for ordinary ComfyUI nodes. |
| `thinking_image` | `IMAGE` | Thought/intermediate images, or a blank placeholder when none are returned. |
| `thinking_process` | `STRING` | Thought text, or `No thinking output returned by the model.` |
| `system_messages` | `STRING` | Response text for compatibility plus routing, migration, retry, validation, availability, and error diagnostics. |
| `generated_image_pipe` | `GENERATED_IMAGE_PIPE` | Original response bytes/MIME, dimensions, order, thought/final status, grounding, API mode, interaction ID, and effective location. Connect it to **Burve Save Generated Image**. |
| `response_text` | `STRING` | Clean model response text without Burve diagnostics. |
| `grounding_metadata_json` | `STRING` | Search queries, steps, sources, citations, usage, and suggestion markup as JSON. |
| `interleaved_content` | `GEMINI_CONTENT_PIPE` | Ordered text/image/thought blocks and artifacts for custom multimodal processing. |
| `interaction_pipe` | `GEMINI_INTERACTION_PIPE` | Provider/model/API-bound state for a later stateful edit. |

`GENERATED_IMAGE_PIPE` version 2 preserves all supported original PNG/JPEG/WebP/HEIC/HEIF bytes. The save node still accepts version 1. Differently sized images remain individually available in `generated_image_pipe` and `interleaved_content`; only dimensions compatible with the first result are placed on the legacy batched `image` output.

### 3. Burve Gemini Reference Pack

Builds role-aware references for the Gemini image nodes.

- **Inputs**: `image1` through `image14` (`IMAGE`, optional), matching `role1` through `role14` (`generic`, `object`, `character`, or `style`) and `label1` through `label14` (`STRING`, optional).
- **Output**: `reference_pack` (`GEMINI_REFERENCE_PIPE`).
- Empty image slots are skipped. Model-specific total and per-role limits are validated by the generator before the request.

### 4. Burve Gemini Input Files

Builds a validated document-context pipe from files inside the ComfyUI input directory.

- **Inputs**: `file1` through `file8` (optional file-upload strings).
- **Output**: `input_files` (`GEMINI_INPUT_FILES_PIPE`).
- Supported extensions are PDF, TXT, Markdown, and CSV. Paths outside the ComfyUI input directory and unsupported formats are rejected.

### 5. Burve List Google Image Models

Performs read-only model discovery for an account/provider/location and compares it with the plugin catalog.

- **Inputs**: `provider` (`aistudio` or `vertex`) and `vertex_location_mode` (`auto` or `configured`).
- **Output**: `model_report` (`STRING`) with provider/location, catalog, lifecycle, and availability diagnostics.
- API keys and credential contents are never included in the report.

### 3.0 migration and privacy

No workflow edit is required for retired preview names. For example, `{"model": "gemini-3.1-flash-image-preview", "resolution": "1K"}` is normalized to `gemini-3.1-flash-image`, with a migration notice in `system_messages`. Retired preview selections keep their old GenerateContent routing for stateless `auto` execution while calling the stable endpoint. Existing selections are otherwise unchanged; only newly created nodes default to Nano Banana 2.1. The node IDs, input names/order, and all nine output positions remain unchanged.

The plugin does not write API keys, prompts, uploaded document/video/image content, or interaction contents to logs. Server-side retention is opt-in through `conversation_mode=stateful`; stateless Interactions explicitly send `store=false`.

### 6. Burve Image Reference Pack
A utility node to bundle multiple images into a single list for the generator node.

*   **Functionality**: Accepts up to 14 individual image inputs and packages them into a format compatible with the `reference_images` input of the **Burve Google Image Gen** node.
*   **Inputs**: `image1` through `image14` (optional).
*   **Outputs**: `images` (A list of images).

### 7. Burve Crop + Mask Load
An interactive image-loader node with crop-region editing and soft mask painting.

*   **Functionality**: Loads an image from the ComfyUI input folder, lets you choose a fixed aspect-ratio crop, move and resize the crop interactively, and build a soft mask with brush, erase, whole-crop fill, and connected-area bucket tools.
*   **Inputs**:
    *   `image`: Select an image from the ComfyUI input directory, or drag an image into the editor to upload it there.
    *   `aspect_ratio`: Crop ratio preset shared across the Burve image-gen-safe set: `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`.
    *   `editor_state_json` (Advanced/internal): Stored UI state used by the custom editor. The frontend manages this automatically.
*   **Outputs**:
    *   `image`: The original source image as a standard `IMAGE`.
    *   `cut_image`: The cropped image region as a standard `IMAGE`.
    *   `image_mask`: A full-size soft `MASK` matching the original image dimensions. `1.0` means painted/masked, `0.0` means clear.
    *   `aspect_ratio`: The selected ratio string, suitable for plugging into `aspect_ratio_override` on compatible Burve image-generation workflows.
    *   `crop_region_pipe`: A `CROP_REGION_PIPE` bundle carrying source size, selected aspect ratio, and pixel crop rect for a future downstream region-aware node.
*   **Editor behavior**:
    *   `List` / `Grid` switches the inline image picker between a filtered text list and thumbnail cards.
    *   `Crop` mode: drag the crop frame to move it, drag corner handles to resize it while preserving ratio.
    *   `Paint` / `Erase` modes: draw soft strokes clipped to the crop area only.
    *   `Bucket` fills the connected unmasked area under the cursor. Existing fully masked pixels act as boundaries, four-connected filling does not cross diagonal walls, and all generated runs are clipped to the current crop.
    *   Drag and drop an image file anywhere inside the custom editor to upload it directly into the ComfyUI input folder.
    *   `Upload` opens a file picker that uses the same in-editor upload flow as drag-and-drop. `Refresh` re-syncs the input folder list without leaving the node.
    *   Mouse wheel zooms around the cursor. Space-drag or middle-mouse drag pans the viewport.
    *   `Fit` resets the viewport. `Reset Crop` restores the centered default crop for the current ratio. `Clear Mask` removes all mask operations.
    *   `Fill Mask` masks the full crop while preserving the ability to erase afterward. `Undo Last` removes the most recent brush, erase, fill, or bucket operation.
    *   `Mask Check` temporarily displays mask strength as grayscale (`black=clear`, `white=fully masked`) without changing the saved state or output mask.
*   **Notes**:
    *   Saved workflows persist the crop box, viewport, brush settings, mask strokes, full fills, and bucket runs through the hidden `editor_state_json` widget.
    *   The selected input image still persists through the hidden native `image` widget, so existing workflows remain compatible.
    *   Changing the selected image resets the editor state when the source file or source dimensions no longer match.
    *   The editor refreshes the input image list in place and keeps the selected image synced when possible.
    *   The image picker mode is remembered globally in the frontend setting `Burve.CropMaskLoad.ImagePickerMode` and defaults to `list`.
    *   The editor keeps crop handles, brush previews, and mask points aligned across graph zoom, browser zoom, HiDPI, and fractional scaling.
    *   The embedded editor detects stale DOM sizing, reflows once, remounts once if needed, and preserves node dimensions and editor state during recovery.
    *   This first release is loader-only and does not accept an upstream `IMAGE` input.

### 8. Burve Crop + Mask Apply
Applies a crop-region replacement back onto a full-size image using a full-size mask.

*   **Functionality**: Takes a base image, a replacement image, the `crop_region_pipe` from **Burve Crop + Mask Load**, and a full-size mask. It resizes the replacement image to the crop rectangle and blends only masked pixels back onto the base image.
*   **Inputs**:
    *   `base_image`: The full-size original image that will receive the edit.
    *   `new_image`: A single-frame image that may be any size. Exact and near-match aspect ratios are accepted and resized to the crop rectangle.
    *   `crop_region_pipe`: The `CROP_REGION_PIPE` output from **Burve Crop + Mask Load**.
    *   `mask`: A full-size soft `MASK`. `1.0` transfers pixels from the placed image, `0.0` keeps the original base image.
    *   `strict_aspect_ratio`: Advanced toggle. When enabled, any aspect ratio mismatch throws a hard error instead of auto-fitting.
*   **Outputs**:
    *   `edited_image`: The base image with only masked pixels inside the crop replaced.
    *   `placed_image_white_bg`: A full-size white canvas with the resized `new_image` filling the crop rectangle.
    *   `masked_placed_image_white_bg`: A full-size white canvas where only masked pixels from the placed image are visible.
    *   `status`: Empty on clean execution, or a warning when a near-match aspect ratio was auto-fit by centered crop.
*   **Notes**:
    *   This node is single-frame only. `base_image`, `new_image`, and `mask` must each have batch size `1`.
    *   Mask values outside the crop rectangle are ignored even if they are nonzero.
    *   Near-match aspect ratios are auto-fit with centered crop when the estimated trim is at most `1.0%`.
    *   Larger aspect mismatches still raise a friendly hard error so the workflow does not silently crop away substantial content.

### 9. Burve Save Generated Image

Saves Gemini outputs while preserving the original response encoding whenever raw artifacts are available.

*   **Inputs**:
    *   `image`: Standard `IMAGE` fallback/batch from either Gemini node.
    *   `filename_prefix`: Output filename prefix; default `Burve`.
    *   `generated_image_pipe` (Optional): `GENERATED_IMAGE_PIPE` v1 or v2 from either Gemini node.
*   **Output**: `saved_files` (`STRING`) listing saved paths, format fallbacks, skipped artifacts, and pipe notes.
*   **Behavior**:
    *   With a v2 pipe, supported PNG, JPEG, WebP, HEIC, and HEIF bytes are written exactly as returned by Google.
    *   Final artifacts are preferred over thought artifacts. If no final exists, available thought artifacts can still be saved.
    *   Mixed-size or non-batched artifacts are saved individually instead of being lost from the legacy `IMAGE` batch.
    *   If raw bytes or a supported MIME type are unavailable, the matching `IMAGE` frame is saved as PNG with normal ComfyUI prompt metadata.

### 10. Burve Debug Gemini Key
A helper node to verify your API key configuration.

*   **Functionality**: Checks if the `GEMINI_API_KEY` environment variable is correctly set and visible to ComfyUI. It displays the status and a masked version of the key.
*   **Outputs**: `info` (Status message).

### 11. Burve Debug Vertex Auth
A helper node to verify your Vertex AI environment configuration.

*   **Functionality**: Reports whether `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_APPLICATION_CREDENTIALS` are present inside ComfyUI, and reminds you that ADC can also come from `gcloud auth application-default login`.
*   **Outputs**: `info` (Status message).

### 12. Burve System Instructions
Selects pre-defined system instructions from a dropdown menu.

*   **Functionality**: Loads system instructions bundled with the installed package and outputs the selected instruction text. Instruction updates arrive with package releases.
*   **Inputs**:
    *   `instruction_name`: Select a system instruction preset.
*   **Outputs**:
    *   `instruction`: The full text of the selected system instruction.

### 13. Burve Variable Injector
Defines variables for use in dynamic prompts.

*   **Functionality**: A utility node to create a dictionary of variable values. Accepts up to 14 string inputs.
*   **Inputs**:
    *   `V1` through `V14` (Optional): String values for variables.
*   **Outputs**:
    *   `variables`: A dictionary of the provided variables.

### 14. Burve Prompt Database
Loads prompts from a database and injects variables.

*   **Functionality**: Selects a prompt from the JSON database bundled with the installed package and replaces placeholders (e.g., `[[name:default]]`) with values from the **Burve Variable Injector**. Prompt updates arrive with package releases.
*   **Inputs**:
    *   `prompt_name`: Select a prompt from the database.
    *   `variables` (Optional): A dictionary of variables from the **Burve Variable Injector** node.
*   **Outputs**:
    *   `compiled_prompt`: The prompt with variables injected.
    *   `raw_prompt`: The original prompt with placeholders.
    *   `title`: The title of the selected prompt.


### 15. Burve Prompt Selector 14

Selects one prompt from a fixed bank of fourteen.

*   **Inputs**:
    *   `prompt_1` through `prompt_14`: Multiline prompt strings.
    *   `index`: Selected prompt number from `1` through `14`.
    *   `error_on_empty`: When enabled, selecting an empty prompt raises an error for the current workflow without affecting unrelated queued work.
*   **Outputs**:
    *   `selected_prompt`: The chosen prompt text.
    *   `selected_index`: The validated selected index.

### 16. Burve Image Info
Reports the dimensions and aspect ratio of an input image.

*   **Functionality**: Reads a standard ComfyUI `IMAGE` tensor, reports its pixel size, and returns a simplified `W:H` aspect ratio string.
*   **Inputs**:
    *   `image`: The input image from a node such as `Load Image`.
*   **Outputs**:
    *   `info`: Human-readable text such as `Size: 1024x768 px` and `Aspect ratio: 4:3`.
    *   `width`: Image width in pixels.
    *   `height`: Image height in pixels.
    *   `aspect_ratio`: Simplified ratio string such as `4:3`, `16:9`, or `1:1`.

### 17. Burve Blind Grid Splitter
Splits an image into a grid of tiles without content analysis.

*   **Functionality**: Slices an input image into a specified number of rows and columns. Useful for processing large images in chunks.
*   **Inputs**:
    *   `image`: The input image to split.
    *   `rows`: Number of horizontal slices (default: 2).
    *   `cols`: Number of vertical slices (default: 2).
    *   `center_crop`: If enabled, centers the grid on the image if dimensions aren't perfectly divisible, otherwise starts from top-left.
*   **Outputs**:
    *   `tiles`: A batch of images containing the resulting grid tiles.

### 18. Burve Character Planner
Builds a reusable base-character prompt bundle for `Burve Google Image Gen`.

*   **Functionality**: Combines curated body, age, gender, race, and appearance controls with optional raw JSON overrides, emits a generation-ready prompt, emits optional face-lock system instructions, and packs ordered reference images with the dedicated face image first.
*   **Scope**: The planner is adult-only and now supports both female and male base characters, plus optional fantasy race details for later outfit swaps and body-shape consistency.
*   **Inputs**:
    *   Identity controls: `gender`, adult-only `age_years` (`18..80`), `race`, inline `custom_race`
    *   Shared body controls: `height_cm`, `weight_kg`, `waist_cm`, `full_hip_cm`, `musculature_tone`, `body_fat`, `pose`
    *   Female-specific controls: `bust_cm`, `underbust_cm`, `body_frame_preset`, `outfit_variant`
    *   Male-specific controls: `male_chest_cm`, `male_body_frame_preset`, `male_outfit_variant`
    *   Appearance controls: `skin_tone`, inline `custom_skin_tone`, `undertone`, `hair_color`, inline `custom_hair_color`, `hair_length`
    *   Basewear controls: shared `outfit_color`
    *   Race extension: optional `race_override_pipe` from **Burve Character Race Details**
    *   Face controls: `use_face_reference`, `face_reference_strength`, and optional `face_reference_image`
    *   Advanced controls: optional `extra_reference_images` (`IMAGE_LIST`) and optional `plan_overrides_json`
*   **Outputs**:
    *   `prompt`: Structured character-planning prompt text for **Burve Google Image Gen**
    *   `system_instructions`: Blank by default, or a stable face-lock instruction when a dedicated face image is enabled and connected
    *   `reference_images`: Ordered `IMAGE_LIST` compatible with **Burve Google Image Gen**
    *   `character_plan_json`: Normalized round-trippable plan JSON
    *   `summary`: Validation/status output including gender, age, race, fantasy traits, ignored gender-specific controls, face-lock state, reference count, and warnings
    *   `character_pipe`: One-cable `CHARACTER_GEN_PIPE` bundle carrying the generation-ready prompt, optional system instructions, packed reference images, and planner metadata
*   **Notes**:
    *   v1 ships with no custom JS or live badges.
    *   Visual feedback is limited to validation errors plus the `summary` output.
    *   The built-in outfit presets stay non-explicit and intentionally minimal for silhouette readability.
    *   Female mode uses the existing bikini-style basewear presets. Male mode uses underwear presets with `base_outfit.top.type = none`.
    *   `character_plan_json` and `summary` are informational/debug outputs. They do not need to be connected to **Burve Google Image Gen**.
    *   `custom_race`, `custom_skin_tone`, and `custom_hair_color` are inline empty override widgets next to the values they replace.
    *   `plan_overrides_json` still has final precedence, but contradictory male/female outfit or underage overrides are rejected.

### 19. Burve Character Race Details
Builds a reusable fantasy race-detail bundle for `Burve Character Planner`.

*   **Functionality**: Produces a `CHARACTER_RACE_PIPE` with optional race-name override plus curated fantasy anatomy traits such as wings, horns, tails, hooves, scales, claws, and related head or limb features.
*   **Inputs**:
    *   `race_name` plus inline `custom_race_name`
    *   `ears` plus `custom_ears`
    *   `horns` plus `custom_horns`
    *   `wings` plus `custom_wings`
    *   `tail` plus `custom_tail`
    *   `legs_feet` plus `custom_legs_feet`
    *   `skin_surface` plus `custom_skin_surface`
    *   `head_features` plus `custom_head_features`
    *   `hands_arms` plus `custom_hands_arms`
    *   `extra_notes`
*   **Outputs**:
    *   `race_override_pipe`: A `CHARACTER_RACE_PIPE` for the planner
    *   `race_override_json`: Debug-friendly JSON view of the same payload
    *   `summary`: Resolved race trait summary
*   **Notes**:
    *   Each dropdown includes `none` as the “do not select” option.
    *   Each adjacent custom text field overrides the dropdown when filled.
    *   Leaving the node disconnected preserves a regular human planner workflow.

## Recommended Character Workflow

Use this chain when you want a reusable base character with optional face anchoring:

1.  Connect an optional dedicated face image to **Burve Character Planner** `face_reference_image`.
2.  If you also want extra non-face references, connect them through **Burve Image Reference Pack** into `extra_reference_images`.
3.  Optional fantasy workflow:
    *   Connect **Burve Character Race Details** `race_override_pipe` into **Burve Character Planner** `race_override_pipe`
4.  Recommended 1-wire workflow:
    *   `character_pipe` -> `character_pipe`
5.  If `character_pipe` is connected, **Burve Google Image Gen** ignores its direct `prompt`, `system_instructions`, and `reference_images` inputs. If an older workflow still shows `A futuristic city` in the prompt widget, clear that saved value or reload the node after updating.
6.  If you want to inspect the normalized plan or warnings, optionally connect `character_plan_json` and `summary` to text/debug nodes. They are not consumed by **Burve Google Image Gen**.
7.  Keep using direct generator inputs only for non-planner workflows.

For fantasy or non-standard characters, use `custom_race`, `custom_hair_color`, and `custom_skin_tone` for quick overrides. Use **Burve Character Race Details** for common non-human anatomy, and keep `plan_overrides_json` for advanced control.

Example `plan_overrides_json` fallback:

```json
{
  "identity": {
    "gender": "male",
    "age_years": 37,
    "age_group": "adult",
    "race": {
      "base": "dragonkin"
    },
    "skin_tone": {
      "base": "green"
    }
  },
  "fantasy_traits": {
    "wings": "dragon_membrane",
    "horns": "swept_back",
    "skin_surface": "light_scales"
  },
  "base_outfit": {
    "top": {
      "type": "none"
    }
  },
  "hair": {
    "scalp_hair": {
      "color": "pink"
    }
  },
  "skin": {
    "texture": {
      "micro_detail": 0.5
    }
  }
}
```

## Installation

1.  Clone this repository into your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/Burve/ComfyUI_Burve_Tools.git
    ```
2.  Install the required dependencies:
    ```bash
    cd ComfyUI_Burve_Tools
    pip install -r requirements.txt
    ```
    (Note: You may need to use the `pip` associated with your ComfyUI python environment, e.g., `python_embeded/python.exe -m pip install ...` if using the portable version).
3.  Use ComfyUI `>=0.4.0`. This is the first tagged release containing the native V3 `VIDEO`/`VideoInput` interface used by the Gemini video socket, along with the required DynamicCombo API.
4.  This release requires `google-genai>=2.13.0,<3` and `pillow-heif>=0.18.0,<2`. Compatibility is checked by required SDK symbols and the Interactions client surface, not by rejecting SDK major version 2.

## Authentication Setup

This project now exposes two separate image-generation nodes:

*   **Burve Google Image Gen** uses AI Studio and reads `GEMINI_API_KEY`.
*   **Burve Google Image Gen (Vertex AI)** uses standard Vertex AI auth with `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and Google credentials / ADC.

As of **August 25, 2026**, this implementation does **not** use a separate Vertex API key path in Python. It uses the documented `google-genai` Vertex client flow.

### AI Studio Setup (`Burve Google Image Gen`)

This node requires a Google Gemini API key in `GEMINI_API_KEY`.

#### How to get an API Key
1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Create a new API key.

#### Setting the Environment Variable

**Windows (PowerShell):**
Run the following command in PowerShell:
```powershell
setx GEMINI_API_KEY "YOUR_REAL_KEY_HERE"
```
*After running this, you must close and restart your terminal and ComfyUI for the change to take effect.*

**macOS (ComfyUI launched from Terminal):**
If you start ComfyUI from Terminal, add the export command to your shell profile. This project reads `GEMINI_API_KEY` from the process environment, and terminal-launched apps inherit that shell environment.
```bash
echo 'export GEMINI_API_KEY="YOUR_REAL_KEY_HERE"' >> ~/.zshrc
source ~/.zshrc
```
Then start ComfyUI from that same terminal session, or open a new terminal after reloading your profile and launch ComfyUI there.

**macOS (Standalone ComfyUI launched from Applications / Finder / Launchpad):**
If you launch the standalone ComfyUI app from Finder, Launchpad, or the Applications folder, do not rely on `~/.zshrc` or `~/.bashrc`. macOS GUI apps do not reliably inherit interactive shell startup files.
```bash
launchctl setenv GEMINI_API_KEY "YOUR_REAL_KEY_HERE"
launchctl getenv GEMINI_API_KEY
```
After setting the variable, fully quit ComfyUI and relaunch it from Applications, Finder, or Launchpad.

`launchctl setenv` is appropriate for the current login session. After logout or restart, you may need to set it again unless you configure a persistent `LaunchAgent` or another system-level environment setup.

**Linux:**
Add the export command to your shell configuration file (for example `~/.bashrc` or `~/.zshrc`):
```bash
export GEMINI_API_KEY="YOUR_REAL_KEY_HERE"
```
Then reload your shell config and restart ComfyUI.

#### Troubleshooting
This node reads only the `GEMINI_API_KEY` process environment variable. If **Burve Google Image Gen** reports that the API key is missing, use **Burve Debug Gemini Key** to inspect what ComfyUI is seeing.

If the debug node shows `GEMINI_API_KEY is NOT set`:

*   Confirm which launch method you are using.
*   On macOS standalone builds, verify the variable with `launchctl getenv GEMINI_API_KEY`.
*   Fully quit and relaunch ComfyUI after changing the environment variable.
*   Check that the key does not contain accidental whitespace or a trailing newline.
*   If you get a tiny black placeholder image plus `No non-thinking image generated.`, the request may have completed without returning a usable image part. Inspect `system_messages` for finish reason or prompt feedback, retry the prompt, and try another supported image model if the response looks text-only or blocked.
*   Version 3.0 keeps the existing node IDs, original input names, and first five output positions. Retired preview selections migrate automatically; newly appended controls use safe defaults.

### Vertex AI Setup (`Burve Google Image Gen (Vertex AI)`)

This node uses standard Vertex AI auth for the Python `google-genai` SDK.

### Before You Start

Before you try the Vertex node, make sure all of the following are true:

*   You have a Google Cloud project.
*   Billing is enabled on that project.
*   The Vertex AI API is enabled on that project.
*   You have local Google credentials available through ADC or a service account.
*   This node does **not** use `GEMINI_API_KEY`.

The current implementation constructs the client like this:

```python
genai.Client(vertexai=True, project=..., location=...)
```

As of **August 25, 2026**, the documented Python path for this node is standard Vertex auth, not a separate Vertex API key flow.

### Step 1: Create or choose a Google Cloud project

1.  Open Google Cloud Console.
2.  Create a new project or select an existing one.
3.  Confirm billing is enabled for that project.
4.  Enable the **Vertex AI API**.

Optional CLI commands:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com
```

### Step 2: Install and initialize `gcloud`

Install the Google Cloud CLI, then initialize it locally:

```bash
gcloud init
gcloud auth login
```

If you are using Cloud Shell instead of a local machine, some local authentication steps may differ. The guidance below is for running ComfyUI on your own machine.

### Step 3: Set up authentication for local ComfyUI use

Use one of these authentication paths.

**Recommended for local testing: Application Default Credentials (ADC) with your user account**

```bash
gcloud auth application-default login
```

**Optional: service-account key file**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/service-account.json"
```

**Advanced: service-account impersonation**

```bash
gcloud auth application-default login --impersonate-service-account=SERVICE_ACCT_EMAIL
```

Notes:

*   The recommended default for local use is `gcloud auth application-default login`.
*   A service-account JSON file is optional, not required.
*   If you use a service account, it must have sufficient Vertex AI access in the selected project.
*   Do not create raw OAuth access tokens manually for this node.

### Step 4: Set environment variables required by this node

This node depends on these environment variables:

*   Required:
    *   `GOOGLE_CLOUD_PROJECT`
    *   `GOOGLE_CLOUD_LOCATION`
*   Optional:
    *   `GOOGLE_APPLICATION_CREDENTIALS` only if you are using a service-account JSON file

Example:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"
```

Important notes:

*   `Burve Google Image Gen (Vertex AI)` does not use `GEMINI_API_KEY`.
*   Vertex auth for this node is Google credentials / ADC plus project and location.
*   `GOOGLE_GENAI_USE_VERTEXAI=true` is supported by the SDK in general, but it is **not required** by this node because the code already initializes the client with `vertexai=True`.
*   Having only `GOOGLE_APPLICATION_CREDENTIALS` is **not enough**. That file authenticates the SDK, but this node still requires `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

### Step 5: Launch ComfyUI so it can see those variables

**If you launch ComfyUI from Terminal**

Export the variables in the same shell session, then start ComfyUI from that same shell.

**If you launch standalone ComfyUI from macOS Applications / Finder / Launchpad**

GUI apps often do not inherit your shell startup files. Set the variables with `launchctl`, then fully quit and relaunch ComfyUI:

```bash
launchctl setenv GOOGLE_CLOUD_PROJECT "your-project-id"
launchctl setenv GOOGLE_CLOUD_LOCATION "global"
launchctl setenv GOOGLE_APPLICATION_CREDENTIALS "/full/path/to/service-account.json"
```

If you are using ADC from `gcloud auth application-default login` and not a service-account JSON, you do not need to set `GOOGLE_APPLICATION_CREDENTIALS`.

**If you launch ComfyUI on Windows**

Use PowerShell `setx` for the project and location, then restart ComfyUI:

```powershell
setx GOOGLE_CLOUD_PROJECT "your-project-id"
setx GOOGLE_CLOUD_LOCATION "global"
```

`setx` affects future processes only. Close and reopen the terminal or app launcher before starting ComfyUI again.

ADC still comes from the Google auth flow such as `gcloud auth application-default login`; these environment variables do not replace authentication by themselves.

### Step 6: Verify inside ComfyUI

1.  Add **Burve Debug Vertex Auth**.
2.  Run it and confirm it reports:
    *   `GOOGLE_CLOUD_PROJECT present: yes`
    *   `GOOGLE_CLOUD_LOCATION present: yes`
    *   `GOOGLE_APPLICATION_CREDENTIALS present: yes` or `no`, depending on your auth path
3.  Add **Burve Google Image Gen (Vertex AI)**.
4.  Use a simple prompt and a default model.
5.  If generation fails, inspect the `system_messages` output.

Important:

*   `GOOGLE_APPLICATION_CREDENTIALS present: no` can still be valid if you authenticated with `gcloud auth application-default login`.
*   The debug node only reports visible environment variables. ADC can still work even when no credential file path is set in the environment.
*   `GOOGLE_APPLICATION_CREDENTIALS present: yes` still does not mean the node is fully configured. You also need `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

### Step 7: Common problems

**Symptom:** `Vertex AI configuration is incomplete`

*   Cause: `GOOGLE_CLOUD_PROJECT` or `GOOGLE_CLOUD_LOCATION` is missing.
*   Fix: Set both variables in the same environment that launches ComfyUI, then restart ComfyUI.

**Symptom:** I set `GOOGLE_APPLICATION_CREDENTIALS`, but the node still says configuration is incomplete

*   Cause: Credentials are configured, but project and/or location are still missing.
*   Fix: Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in the same environment that launches ComfyUI.

**Symptom:** ComfyUI cannot see the variables

*   Cause: ComfyUI was launched from a different shell, app session, or GUI context.
*   Fix: Re-launch ComfyUI from the shell where you exported the variables, or use `launchctl setenv` on macOS standalone builds.

**Symptom:** Auth or permission errors after project and location are present

*   Cause: ADC is missing, expired, using the wrong account, or the account does not have sufficient Vertex AI permissions.
*   Fix: Re-run `gcloud auth application-default login`, verify the active Google account, or use a service account with the required Vertex AI access.

**Symptom:** Requests fail even though auth looks correct

*   Cause: Billing is disabled or the Vertex AI API is not enabled for the selected project.
*   Fix: Enable billing and enable the Vertex AI API in that same Google Cloud project.

**Symptom:** Model or region errors

*   Cause: The chosen location may not support the model or your project setup.
*   Fix: Try `global` first and verify model availability for your project and region.

**Symptom:** The image node stays active for a long time and appears stuck

*   Cause: The Gemini request is slow, retrying transient failures, or not returning within the configured timeout budget.
*   Fix: Inspect `system_messages` after the node finishes. For `gemini-nano-banana-2.1` or `gemini-3.1-flash-image`, try `global`, lower `resolution`, `MINIMAL` `thinking_level`, or a higher advanced `request_timeout_seconds`.
*   The node now uses a bounded timeout and returns an explicit timeout or retry-exhaustion error instead of waiting indefinitely.

**Symptom:** Nano Banana 2.1 or `gemini-3.1-flash-image` is slow

*   Cause: These models support Thinking and use the `global` Vertex location. Higher resolution and higher thinking levels can increase latency; Nano Banana 2.1 defaults to `MEDIUM` thinking.
*   Fix: Start with `global`, `1K` or `2K`, and `MINIMAL` thinking when you need a more responsive workflow.

### What this node actually uses

The node currently uses:

```python
genai.Client(vertexai=True, project=..., location=...)
```

That means:

*   `Burve Google Image Gen` uses `GEMINI_API_KEY`.
*   `Burve Google Image Gen (Vertex AI)` does **not** use `GEMINI_API_KEY`.
*   Vertex auth is Google credentials / ADC plus project and location.
*   `GOOGLE_GENAI_USE_VERTEXAI` is optional for the SDK in general, but not required by this node's current implementation.
*   Nano Banana 2.1 and the catalog's Gemini 3.x image models use the `global` Vertex endpoint.

Official references:

*   [Gemini image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)
*   [Nano Banana 2.1 model](https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1)
*   [Nano Banana 2.1 on Google Cloud](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/nano-banana-2-1)
*   [Gemini GenerateContent image guide](https://ai.google.dev/gemini-api/docs/generate-content/image-generation)
*   [Interactions response formats](https://ai.google.dev/api/interactions-api#ImageResponseFormat)
*   [Gemini model deprecations](https://ai.google.dev/gemini-api/docs/deprecations)
*   [October 6, 2026 release notes](https://ai.google.dev/gemini-api/docs/changelog#october-6-2026)
*   [Gemini 2.5 Flash Image lifecycle on Google Cloud](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash-image)
*   [Interactions API overview](https://ai.google.dev/gemini-api/docs/interactions-overview)
*   [Google Gen AI Python SDK overview](https://googleapis.github.io/python-genai/)
*   [google-genai on PyPI](https://pypi.org/project/google-genai/)
*   [Vertex AI authentication](https://cloud.google.com/vertex-ai/docs/authentication)
*   [Vertex project/environment setup](https://docs.cloud.google.com/vertex-ai/docs/start/cloud-environment)
