# ComfyUI Burve Tools

A collection of custom nodes for ComfyUI, focusing on integration with Google's Gemini AI models for image generation and more.

## Nodes

### 1. Burve Google Image Gen
This is the main node for generating images using Google's Gemini models.

*   **Functionality**: Generates images based on text prompts, with support for system instructions and reference images.
*   **Inputs**:
    *   `prompt`: The description of the image you want to generate.
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
    *   `system_instructions` (Optional): Advanced instructions to guide the model's behavior.
    *   `reference_images` (Optional): A list of images to use as context/reference for the generation. Use the **Burve Image Ref Pack** node or the **Burve Character Planner** node to create this list.
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

*   **Functionality**: Combines curated body/appearance controls with optional raw JSON overrides, emits a generation-ready prompt, emits optional face-lock system instructions, and packs ordered reference images with the dedicated face image first.
*   **Scope**: v1 is intentionally focused on `adult_female_photorealistic` base-character planning for later outfit swaps and body-shape consistency.
*   **Inputs**:
    *   Core body controls: `height_cm`, `weight_kg`, `bust_cm`, `underbust_cm`, `waist_cm`, `full_hip_cm`
    *   Appearance controls: `body_frame_preset`, `skin_tone`, `undertone`, `hair_color`, `hair_length`, `musculature_tone`, `body_fat`, `pose`
    *   Basewear controls: `outfit_variant` (`classic_triangle`, `soft_scoop`, `narrow_triangle`, `halter_contour`) and `outfit_color` (`neutral_gray`, `soft_taupe`, `muted_black`)
    *   Face controls: `use_face_reference`, `face_reference_strength`, and optional `face_reference_image`
    *   Advanced controls: optional `extra_reference_images` (`IMAGE_LIST`) and optional `plan_overrides_json`
*   **Outputs**:
    *   `prompt`: Structured character-planning prompt text for **Burve Google Image Gen**
    *   `system_instructions`: Blank by default, or a stable face-lock instruction when a dedicated face image is enabled and connected
    *   `reference_images`: Ordered `IMAGE_LIST` compatible with **Burve Google Image Gen**
    *   `character_plan_json`: Normalized round-trippable plan JSON
    *   `summary`: Validation/status output including face-lock state, resolved outfit variant, reference count, and warnings
*   **Notes**:
    *   v1 ships with no custom JS, live badges, or `WEB_DIRECTORY` frontend extension.
    *   Visual feedback is limited to validation errors plus the `summary` output.
    *   The built-in outfit presets stay non-explicit and intentionally minimal for silhouette readability.

## Recommended Character Workflow

Use this chain when you want a reusable base character with optional face anchoring:

1.  Connect an optional dedicated face image to **Burve Character Planner** `face_reference_image`.
2.  If you also want extra non-face references, connect them through **Burve Image Ref Pack** into `extra_reference_images`.
3.  Connect planner outputs directly into **Burve Google Image Gen**:
    *   `prompt` -> `prompt`
    *   `system_instructions` -> `system_instructions`
    *   `reference_images` -> `reference_images`
4.  Keep using **Burve Google Image Gen** directly for legacy workflows if you do not need the planner.

The planner does not change the public API of **Burve Google Image Gen**. It only prepares compatible inputs for the existing node.

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
