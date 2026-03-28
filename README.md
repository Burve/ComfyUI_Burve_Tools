# ComfyUI Burve Tools

A collection of custom nodes for ComfyUI, focusing on integration with Google's Gemini AI models for image generation and more.

## Nodes

### 1. Burve Google Image Gen
This is the AI Studio version of the main image-generation node.

*   **Functionality**: Generates images based on text prompts, with support for system instructions, reference images, and an optional external aspect-ratio override.
*   **Inputs**:
    *   `prompt`: The description of the image you want to generate. When `character_pipe` is connected, this direct field is ignored.
    *   `model`: A DynamicCombo model selector. The selected model controls which nested fields are shown.
    *   Common fields outside the DynamicCombo: `seed` with `control_after_generate`, `system_instructions`, `reference_images`, `character_pipe`, `aspect_ratio_override`.
    *   `gemini-2.5-flash-image` fields: `aspect_ratio`
    *   `gemini-3-pro-image-preview` fields: `aspect_ratio`, `resolution`, `search_mode`
    *   `gemini-3-pro-image-preview` may still return thought text and thought images in the existing outputs, but it does not expose user-configurable thinking controls.
    *   `gemini-3.1-flash-image-preview` fields: `aspect_ratio`, `resolution`, `search_mode`, `thinking_level`, `include_thoughts`, `output_mime_type`, `prominent_people`
    *   `resolution` values:
        `gemini-3-pro-image-preview`: `1K`, `2K`, `4K`
        `gemini-3.1-flash-image-preview`: `512`, `1K`, `2K`, `4K`
    *   `search_mode` values:
        `gemini-3-pro-image-preview`: `off`, `web`
        `gemini-3.1-flash-image-preview`: `off`, `web`, `image`, `web+image`
    *   `thinking_level` values:
        `gemini-3.1-flash-image-preview`: `MINIMAL`, `HIGH`
    *   `output_mime_type` values for `gemini-3.1-flash-image-preview`: `image/png`, `image/jpeg`, `image/webp`
    *   `prominent_people` values for `gemini-3.1-flash-image-preview`: `allow`, `block`
    *   `seed`: Seed for reproducibility.
    *   `system_instructions` (Optional): Advanced instructions to guide the model's behavior. Ignored when `character_pipe` is connected.
    *   `reference_images` (Optional): A list of images to use as context/reference for the generation. Use the **Burve Image Ref Pack** node or the **Burve Character Planner** node to create this list. Ignored when `character_pipe` is connected.
    *   `character_pipe` (Optional): A one-cable `CHARACTER_GEN_PIPE` bundle from **Burve Character Planner**. When connected, the node becomes pipe-authoritative and uses the planner's `prompt`, `system_instructions`, and `reference_images`.
    *   `aspect_ratio_override` (Optional): A plain `W:H` ratio string socket for nodes like **Burve Crop + Mask Load** or **Burve Image Info**. When connected or manually set, it overrides the DynamicCombo model's internal `aspect_ratio` field.
    *   Advanced optional request controls:
        `request_timeout_seconds`: Overall Gemini request timeout budget in seconds. Default: `120`.
        `retry_attempts`: Total attempts for transient request failures. Default: `5`. `1` disables retries.
*   **Aspect-ratio override behavior**:
    *   If `aspect_ratio_override` is empty, the node uses the `aspect_ratio` selected inside the DynamicCombo model UI.
    *   If `aspect_ratio_override` is present and the selected model supports it, the override is used for the request.
    *   If `aspect_ratio_override` is present but the selected model does not support it, the node throws an error before any Gemini request is sent.
*   **Outputs**:
    *   `image`: The generated image(s).
    *   `thinking_image`: Any thought-stage image(s) returned by the model.
    *   `thinking_process`: Reasoning/thought text. If the model returns none, the node outputs `No thinking output returned by the model.`
    *   `system_messages`: Error messages or status updates. If the model returns none, the node outputs `No system messages.` Requests that exceed the timeout budget now fail with a bounded timeout error instead of waiting indefinitely.

### 2. Burve Google Image Gen (Vertex AI)
This is the Vertex AI version of the same image-generation node.

*   **Functionality**: Uses the same generation logic, DynamicCombo model UI, inputs, and outputs as **Burve Google Image Gen**, but authenticates through Vertex AI instead of AI Studio.
*   **Inputs**: Identical to **Burve Google Image Gen**.
*   **Outputs**: Identical to **Burve Google Image Gen**.
*   **Authentication**: Uses `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and Google credentials / ADC. It does not use `GEMINI_API_KEY`.

### 3. Burve Image Ref Pack
A utility node to bundle multiple images into a single list for the generator node.

*   **Functionality**: Accepts up to 14 individual image inputs and packages them into a format compatible with the `reference_images` input of the **Burve Google Image Gen** node.
*   **Inputs**: `image1` through `image14` (optional).
*   **Outputs**: `images` (A list of images).

### 4. Burve Crop + Mask Load
An interactive image-loader node with crop-region editing and soft mask painting.

*   **Functionality**: Loads an image from the ComfyUI input folder, lets you choose a fixed aspect-ratio crop, move and resize the crop interactively, and paint a soft mask that stays constrained to the crop area.
*   **Inputs**:
    *   `image`: Select an image from the ComfyUI input directory.
    *   `aspect_ratio`: Crop ratio preset shared across the Burve image-gen-safe set: `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`.
    *   `editor_state_json` (Advanced/internal): Stored UI state used by the custom editor. The frontend manages this automatically.
*   **Outputs**:
    *   `image`: The original source image as a standard `IMAGE`.
    *   `cut_image`: The cropped image region as a standard `IMAGE`.
    *   `image_mask`: A full-size soft `MASK` matching the original image dimensions. `1.0` means painted/masked, `0.0` means clear.
    *   `aspect_ratio`: The selected ratio string, suitable for plugging into `aspect_ratio_override` on compatible Burve image-generation workflows.
    *   `crop_region_pipe`: A `CROP_REGION_PIPE` bundle carrying source size, selected aspect ratio, and pixel crop rect for a future downstream region-aware node.
*   **Editor behavior**:
    *   `Crop` mode: drag the crop frame to move it, drag corner handles to resize it while preserving ratio.
    *   `Paint` / `Erase` modes: draw soft strokes clipped to the crop area only.
    *   Mouse wheel zooms around the cursor. Space-drag or middle-mouse drag pans the viewport.
    *   `Fit` resets the viewport. `Reset Crop` restores the centered default crop for the current ratio. `Clear Mask` removes all painted strokes.
*   **Notes**:
    *   Saved workflows persist the crop box, viewport, brush settings, and mask strokes through the hidden `editor_state_json` widget.
    *   Changing the selected image resets the editor state when the source file or source dimensions no longer match.
    *   The editor keeps crop handles, brush previews, and paint/erase points aligned across graph zoom, browser zoom, HiDPI, and fractional scaling.
    *   This first release is loader-only and does not accept an upstream `IMAGE` input.

### 5. Burve Crop + Mask Apply
Applies a crop-region replacement back onto a full-size image using a full-size mask.

*   **Functionality**: Takes a base image, a replacement image, the `crop_region_pipe` from **Burve Crop + Mask Load**, and a full-size mask. It validates the replacement image aspect ratio, resizes it to the crop rectangle, and blends only masked pixels back onto the base image.
*   **Inputs**:
    *   `base_image`: The full-size original image that will receive the edit.
    *   `new_image`: A single-frame image whose aspect ratio must match the crop pipe ratio. It may be any size and will always be resized to the crop rectangle.
    *   `crop_region_pipe`: The `CROP_REGION_PIPE` output from **Burve Crop + Mask Load**.
    *   `mask`: A full-size soft `MASK`. `1.0` transfers pixels from the placed image, `0.0` keeps the original base image.
*   **Outputs**:
    *   `edited_image`: The base image with only masked pixels inside the crop replaced.
    *   `placed_image_white_bg`: A full-size white canvas with the resized `new_image` filling the crop rectangle.
    *   `masked_placed_image_white_bg`: A full-size white canvas where only masked pixels from the placed image are visible.
*   **Notes**:
    *   This node is single-frame only. `base_image`, `new_image`, and `mask` must each have batch size `1`.
    *   Mask values outside the crop rectangle are ignored even if they are nonzero.
    *   A mismatched `new_image` aspect ratio throws a hard error instead of auto-cropping.

### 6. Burve Debug Gemini Key
A helper node to verify your API key configuration.

*   **Functionality**: Checks if the `GEMINI_API_KEY` environment variable is correctly set and visible to ComfyUI. It displays the status and a masked version of the key.
*   **Outputs**: `info` (Status message).

### 7. Burve Debug Vertex Auth
A helper node to verify your Vertex AI environment configuration.

*   **Functionality**: Reports whether `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_APPLICATION_CREDENTIALS` are present inside ComfyUI, and reminds you that ADC can also come from `gcloud auth application-default login`.
*   **Outputs**: `info` (Status message).

### 8. Burve System Instructions
Selects pre-defined system instructions from a dropdown menu.

*   **Functionality**: Loads system instructions from a JSON file (which auto-updates from GitHub) and outputs the selected instruction text.
*   **Inputs**:
    *   `instruction_name`: Select a system instruction preset.
*   **Outputs**:
    *   `instruction`: The full text of the selected system instruction.

### 9. Burve Variable Injector
Defines variables for use in dynamic prompts.

*   **Functionality**: A utility node to create a dictionary of variable values. Accepts up to 14 string inputs.
*   **Inputs**:
    *   `V1` through `V14` (Optional): String values for variables.
*   **Outputs**:
    *   `variables`: A dictionary of the provided variables.

### 10. Burve Prompt Database
Loads prompts from a database and injects variables.

*   **Functionality**: Selects a prompt from a JSON database (auto-updating) and replaces placeholders (e.g., `[[name:default]]`) with values from the **Burve Variable Injector**.
*   **Inputs**:
    *   `prompt_name`: Select a prompt from the database.
    *   `variables` (Optional): A dictionary of variables from the **Burve Variable Injector** node.
*   **Outputs**:
    *   `compiled_prompt`: The prompt with variables injected.
    *   `raw_prompt`: The original prompt with placeholders.
    *   `title`: The title of the selected prompt.


### 11. Burve Image Info
Reports the dimensions and aspect ratio of an input image.

*   **Functionality**: Reads a standard ComfyUI `IMAGE` tensor, reports its pixel size, and returns a simplified `W:H` aspect ratio string.
*   **Inputs**:
    *   `image`: The input image from a node such as `Load Image`.
*   **Outputs**:
    *   `info`: Human-readable text such as `Size: 1024x768 px` and `Aspect ratio: 4:3`.
    *   `width`: Image width in pixels.
    *   `height`: Image height in pixels.
    *   `aspect_ratio`: Simplified ratio string such as `4:3`, `16:9`, or `1:1`.

### 12. Burve Blind Grid Splitter
Splits an image into a grid of tiles without content analysis.

*   **Functionality**: Slices an input image into a specified number of rows and columns. Useful for processing large images in chunks.
*   **Inputs**:
    *   `image`: The input image to split.
    *   `rows`: Number of horizontal slices (default: 2).
    *   `cols`: Number of vertical slices (default: 2).
    *   `center_crop`: If enabled, centers the grid on the image if dimensions aren't perfectly divisible, otherwise starts from top-left.
*   **Outputs**:
    *   `tiles`: A batch of images containing the resulting grid tiles.

### 13. Burve Character Planner
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

### 14. Burve Character Race Details
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
2.  If you also want extra non-face references, connect them through **Burve Image Ref Pack** into `extra_reference_images`.
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
3.  Use a DynamicCombo-capable ComfyUI build. This release now loads through V3 `comfy_entrypoint()` and requires `IO.DynamicCombo.Input(...)`.
4.  `google-genai>=1.68.0,<2` is required for Gemini image search, documented thinking config, and `prominent_people` support.

## Authentication Setup

This project now exposes two separate image-generation nodes:

*   **Burve Google Image Gen** uses AI Studio and reads `GEMINI_API_KEY`.
*   **Burve Google Image Gen (Vertex AI)** uses standard Vertex AI auth with `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and Google credentials / ADC.

As of **March 12, 2026**, this implementation does **not** use a separate Vertex API key path in Python. It uses the documented `google-genai` Vertex client flow.

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
*   This major release removes the old Gemini legacy widgets. Saved workflows using the previous static Gemini parameter list will not retain those removed widget values.

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

As of **March 13, 2026**, the documented Python path for this node is standard Vertex auth, not a separate Vertex API key flow.

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
*   Fix: Inspect `system_messages` after the node finishes. If you are using `gemini-3.1-flash-image-preview`, try `global`, lower `resolution`, `MINIMAL` `thinking_level`, or a higher advanced `request_timeout_seconds`.
*   The node now uses a bounded timeout and returns an explicit timeout or retry-exhaustion error instead of waiting indefinitely.

**Symptom:** `gemini-3.1-flash-image-preview` is slow

*   Cause: This model is still in Preview on Vertex AI, supports Thinking, and uses the `global` location. Higher resolution and higher thinking levels can increase latency.
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
*   `gemini-3.1-flash-image-preview` is documented by Google as a Preview model and global-only on Vertex AI as of March 25, 2026.

Official references:

*   Google Gen AI Python SDK overview: https://googleapis.github.io/python-genai/
*   Google Gen AI Python `Client` reference: https://googleapis.github.io/python-genai/genai.html
*   Vertex AI authentication: https://cloud.google.com/vertex-ai/docs/authentication
*   Vertex project/environment setup: https://docs.cloud.google.com/vertex-ai/docs/start/cloud-environment
