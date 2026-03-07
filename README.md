# ComfyUI Burve Tools

A collection of custom nodes for ComfyUI, focusing on integration with Google's Gemini AI models for image generation and more.

## Nodes

### 1. Burve Google Image Gen
This is the main node for generating images using Google's Gemini models.

*   **Functionality**: Generates images based on text prompts, with support for system instructions and reference images.
*   **Inputs**:
    *   `prompt`: The description of the image you want to generate. When `character_pipe` is connected, this direct field is ignored.
    *   `model`: Select between `gemini-3.1-flash-image-preview`, `gemini-3-pro-image-preview`, and `gemini-2.5-flash-image`.
    *   `resolution`: Output resolution (`0.5K`, `1K`, `2K`, `4K`). (`0.5K` is for 3.1 Flash Image.)
    *   `aspect_ratio`: Desired aspect ratio. Includes extra 3.1 Flash Image ratios (`1:4`, `4:1`, `1:8`, `8:1`) plus standard ratios.
    *   `seed`: Seed for reproducibility.
    *   `search_mode`: Preferred search control (`legacy_toggles`, `off`, `web`, `image`, `web+image`).
    *   `thinking_mode`: Preferred thinking control (`legacy_toggle`, `minimal`, `high`) for `gemini-3.1-flash-image-preview`.
    *   `(legacy, optional) enable_google_search`: Enables Google Search grounding when supported by the selected model.
    *   `(legacy, optional) enable_image_search`: Enables Google Image Search grounding for `gemini-3.1-flash-image-preview`.
    *   `(legacy, optional) enable_thinking_mode`: Legacy toggle for 3.1 Flash Image reasoning intensity (`on` = high, `off` = minimal + hidden thoughts).
    *   Legacy compatibility: if `search_mode` is `legacy_toggles`, the node uses `(legacy) enable_google_search` + `(legacy) enable_image_search`; if `thinking_mode` is `legacy_toggle`, it uses `(legacy) enable_thinking_mode`.
    *   `system_instructions` (Optional): Advanced instructions to guide the model's behavior. Ignored when `character_pipe` is connected.
    *   `reference_images` (Optional): A list of images to use as context/reference for the generation. Use the **Burve Image Ref Pack** node or the **Burve Character Planner** node to create this list. Ignored when `character_pipe` is connected.
    *   `character_pipe` (Optional): A one-cable `CHARACTER_GEN_PIPE` bundle from **Burve Character Planner**. When connected, the node becomes pipe-authoritative and uses the planner's `prompt`, `system_instructions`, and `reference_images`.
*   **Outputs**:
    *   `image`: The generated image(s).
    *   `thinking_image`: Any thought-stage image(s) returned by the model.
    *   `thinking_process`: Reasoning/thought text (when returned by the model).
    *   `system_messages`: Error messages or status updates (e.g., if the API key is missing).

### 2. Burve Image Ref Pack
A utility node to bundle multiple images into a single list for the generator node.

*   **Functionality**: Accepts up to 14 individual image inputs and packages them into a format compatible with the `reference_images` input of the **Burve Google Image Gen** node.
*   **Inputs**: `image1` through `image14` (optional).
*   **Outputs**: `images` (A list of images).

### 3. Burve Debug Gemini Key
A helper node to verify your API key configuration.

*   **Functionality**: Checks if the `GEMINI_API_KEY` environment variable is correctly set and visible to ComfyUI. It displays the status and a masked version of the key.
*   **Outputs**: `info` (Status message).

### 4. Burve System Instructions
Selects pre-defined system instructions from a dropdown menu.

*   **Functionality**: Loads system instructions from a JSON file (which auto-updates from GitHub) and outputs the selected instruction text.
*   **Inputs**:
    *   `instruction_name`: Select a system instruction preset.
*   **Outputs**:
    *   `instruction`: The full text of the selected system instruction.

### 5. Burve Variable Injector
Defines variables for use in dynamic prompts.

*   **Functionality**: A utility node to create a dictionary of variable values. Accepts up to 14 string inputs.
*   **Inputs**:
    *   `V1` through `V14` (Optional): String values for variables.
*   **Outputs**:
    *   `variables`: A dictionary of the provided variables.

### 6. Burve Prompt Database
Loads prompts from a database and injects variables.

*   **Functionality**: Selects a prompt from a JSON database (auto-updating) and replaces placeholders (e.g., `[[name:default]]`) with values from the **Burve Variable Injector**.
*   **Inputs**:
    *   `prompt_name`: Select a prompt from the database.
    *   `variables` (Optional): A dictionary of variables from the **Burve Variable Injector** node.
*   **Outputs**:
    *   `compiled_prompt`: The prompt with variables injected.
    *   `raw_prompt`: The original prompt with placeholders.
    *   `title`: The title of the selected prompt.


### 7. Burve Blind Grid Splitter
Splits an image into a grid of tiles without content analysis.

*   **Functionality**: Slices an input image into a specified number of rows and columns. Useful for processing large images in chunks.
*   **Inputs**:
    *   `image`: The input image to split.
    *   `rows`: Number of horizontal slices (default: 2).
    *   `cols`: Number of vertical slices (default: 2).
    *   `center_crop`: If enabled, centers the grid on the image if dimensions aren't perfectly divisible, otherwise starts from top-left.
*   **Outputs**:
    *   `tiles`: A batch of images containing the resulting grid tiles.

### 8. Burve Character Planner
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
    *   v1 ships with no custom JS, live badges, or `WEB_DIRECTORY` frontend extension.
    *   Visual feedback is limited to validation errors plus the `summary` output.
    *   The built-in outfit presets stay non-explicit and intentionally minimal for silhouette readability.
    *   Female mode uses the existing bikini-style basewear presets. Male mode uses underwear presets with `base_outfit.top.type = none`.
    *   `character_plan_json` and `summary` are informational/debug outputs. They do not need to be connected to **Burve Google Image Gen**.
    *   `custom_race`, `custom_skin_tone`, and `custom_hair_color` are inline empty override widgets next to the values they replace.
    *   `plan_overrides_json` still has final precedence, but contradictory male/female outfit or underage overrides are rejected.

### 9. Burve Character Race Details
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

## Google API Key Setup

This project requires a Google Gemini API key. You must set it as an environment variable named `GEMINI_API_KEY`.

### How to get an API Key
1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Create a new API key.

### Setting the Environment Variable

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

### Troubleshooting
This project reads only the `GEMINI_API_KEY` process environment variable. If the **Burve Google Image Gen** node reports that the API key is missing, use the **Burve Debug Gemini Key** node to inspect what ComfyUI is seeing.

If the debug node shows `GEMINI_API_KEY is NOT set`:

*   Confirm which launch method you are using.
*   On macOS standalone builds, verify the variable with `launchctl getenv GEMINI_API_KEY`.
*   Fully quit and relaunch ComfyUI after changing the environment variable.
*   Check that the key does not contain accidental whitespace or a trailing newline.
