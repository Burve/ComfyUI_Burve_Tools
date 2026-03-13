import torch
import numpy as np
from PIL import Image
import os
import io
from google import genai
from google.genai import types
import json
import urllib.request
import ssl
import hashlib

try:
    from .character_planner import (
        build_character_plan,
        CHARACTER_RACE_PIPE_KIND,
        CHARACTER_RACE_PIPE_VERSION,
        RACE_TRAIT_KEYS,
    )
except ImportError:
    from character_planner import (
        build_character_plan,
        CHARACTER_RACE_PIPE_KIND,
        CHARACTER_RACE_PIPE_VERSION,
        RACE_TRAIT_KEYS,
    )

# --- Auto-update logic for system instructions ---
INSTRUCTIONS_FILE = os.path.join(os.path.dirname(__file__), "system_instructions.json")
INSTRUCTIONS_URL = "https://raw.githubusercontent.com/Burve/ComfyUI_Burve_Tools/main/system_instructions.json"
CHARACTER_GEN_PIPE_KIND = "burve.character_gen_pipe"
CHARACTER_GEN_PIPE_VERSION = 1
CHARACTER_PIPE_OVERRIDE_NOTE = (
    "character_pipe is connected; ignoring direct prompt, system_instructions, and "
    "reference_images inputs."
)
RACE_OPTIONS = [
    "human",
    "elf",
    "dark_elf",
    "orc",
    "angel",
    "demon",
    "tiefling",
    "dragonkin",
    "catfolk",
    "wolfkin",
    "satyr",
    "merfolk",
    "fairy",
    "undead",
]
RACE_NAME_OPTIONS = ["none"] + RACE_OPTIONS[1:]
RACE_DETAIL_OPTIONS = {
    "ears": ["none", "pointed_elf", "long_fae", "finned", "feline", "canine", "bovine", "caprine"],
    "horns": ["none", "small_curved", "ram", "antlers", "straight_spire", "swept_back", "crown_horns", "broken_horn"],
    "wings": ["none", "small_feathered", "large_feathered", "bat_leather", "dragon_membrane", "insect_iridescent", "spectral"],
    "tail": ["none", "feline", "canine", "fox", "bovine", "equine", "reptilian", "draconic", "spade_demon", "mer_tail"],
    "legs_feet": ["none", "digitigrade_feline", "digitigrade_canine", "cloven_hoof", "bird_talon", "reptilian_claw", "serpentine_lower_body"],
    "skin_surface": ["none", "light_scales", "heavy_scales", "fur_patches", "full_fur", "bark_texture", "stone_texture", "chitin", "glowing_markings"],
    "head_features": ["none", "animal_muzzle", "full_animal_head", "avian_beak", "tusks", "fangs", "gills", "third_eye", "crest"],
    "hands_arms": ["none", "claws", "talons", "webbed", "scaled_hands", "wing_arms"],
}

def load_instructions():
    if not os.path.exists(INSTRUCTIONS_FILE):
        return {"version": 0, "presets": []}
    try:
        with open(INSTRUCTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[BurveTools] Error loading instructions: {e}")
        return {"version": 0, "presets": []}

def update_instructions():
    print("[BurveTools] Checking for instruction updates...")
    current_data = load_instructions()
    current_version = current_data.get("version", 0)

    try:
        # Create a context that doesn't verify certificates if needed (optional, but good for some envs)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(INSTRUCTIONS_URL, context=ctx, timeout=5) as response:
            if response.status == 200:
                remote_data = json.loads(response.read().decode('utf-8'))
                remote_version = remote_data.get("version", 0)
                
                if remote_version > current_version:
                    print(f"[BurveTools] Updating instructions from v{current_version} to v{remote_version}")
                    with open(INSTRUCTIONS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(remote_data, f, indent=2)
                else:
                    print("[BurveTools] Instructions are up to date.")
    except Exception as e:
        print(f"[BurveTools] Update check failed: {e}")

# Run update on module load
update_instructions()

class BurveGoogleImageGenBase:
    PROVIDER_ID = "base"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (
                    ["gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview", "gemini-2.5-flash-image"],
                    {"default": "gemini-2.5-flash-image"},
                ),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "aspect_ratio": (
                    [
                        "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1",
                        "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
                    ],
                    {"default": "1:1"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "search_mode": (
                    ["legacy_toggles", "off", "web", "image", "web+image"],
                    {"default": "legacy_toggles"},
                ),
                "thinking_mode": (
                    ["legacy_toggle", "minimal", "high"],
                    {"default": "legacy_toggle"},
                ),
            },
            "optional": {
                "enable_google_search": ("BOOLEAN", {"default": False}),
                "enable_image_search": ("BOOLEAN", {"default": False}),
                "enable_thinking_mode": ("BOOLEAN", {"default": True}),
                "system_instructions": ("STRING", {"multiline": True, "default": ""}),
                "reference_images": ("IMAGE_LIST",),
                "character_pipe": ("CHARACTER_GEN_PIPE",),
            },
        }

    # normal images, thinking images, thinking text, system messages
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "thinking_image", "thinking_process", "system_messages")
    FUNCTION = "generate_image"
    CATEGORY = "BurveTools"

    def _get_provider_auth_error(self):
        raise NotImplementedError

    def _build_client(self):
        raise NotImplementedError

    def _resolve_search_mode(self, search_mode, enable_google_search, enable_image_search):
        if search_mode == "off":
            return False, False
        if search_mode == "web":
            return True, False
        if search_mode == "image":
            return False, True
        if search_mode == "web+image":
            return True, True
        return bool(enable_google_search), bool(enable_image_search)

    def _resolve_thinking_mode(self, thinking_mode, enable_thinking_mode):
        if thinking_mode == "minimal":
            return "minimal", False
        if thinking_mode == "high":
            return "high", True
        if enable_thinking_mode:
            return "high", True
        return "minimal", False

    def _blank_result(self, system_messages):
        blank = torch.zeros((1, 64, 64, 3))
        return (blank, blank.clone(), "", system_messages)

    def _is_non_empty_text(self, value):
        return isinstance(value, str) and bool(value.strip())

    def _has_reference_images(self, reference_images):
        if reference_images is None:
            return False

        try:
            for img_tensor in reference_images:
                if (
                    isinstance(img_tensor, torch.Tensor)
                    and img_tensor.ndim >= 4
                    and img_tensor.shape[0] > 0
                ):
                    return True
        except TypeError:
            return False

        return False

    def _normalize_character_pipe(self, character_pipe):
        if character_pipe is None:
            return None

        if not isinstance(character_pipe, dict):
            raise ValueError("Invalid character_pipe: expected a dict payload.")

        if character_pipe.get("kind") != CHARACTER_GEN_PIPE_KIND:
            raise ValueError("Invalid character_pipe: unexpected kind.")

        if character_pipe.get("version") != CHARACTER_GEN_PIPE_VERSION:
            raise ValueError("Invalid character_pipe: unsupported version.")

        if "prompt" not in character_pipe:
            raise ValueError("Invalid character_pipe: missing required key 'prompt'.")

        if not isinstance(character_pipe["prompt"], str):
            raise ValueError("Invalid character_pipe: prompt must be a string.")

        pipe_system_instructions = character_pipe.get("system_instructions", "")
        if not isinstance(pipe_system_instructions, str):
            raise ValueError("Invalid character_pipe: system_instructions must be a string.")

        pipe_character_plan_json = character_pipe.get("character_plan_json", "")
        if not isinstance(pipe_character_plan_json, str):
            raise ValueError("Invalid character_pipe: character_plan_json must be a string.")

        pipe_summary = character_pipe.get("summary", "")
        if not isinstance(pipe_summary, str):
            raise ValueError("Invalid character_pipe: summary must be a string.")

        pipe_reference_images = character_pipe.get("reference_images")
        if pipe_reference_images is None:
            normalized_reference_images = None
        elif isinstance(pipe_reference_images, tuple):
            normalized_reference_images = list(pipe_reference_images)
        elif isinstance(pipe_reference_images, list):
            normalized_reference_images = pipe_reference_images
        else:
            raise ValueError("Invalid character_pipe: reference_images must be a list.")

        return {
            "kind": CHARACTER_GEN_PIPE_KIND,
            "version": CHARACTER_GEN_PIPE_VERSION,
            "prompt": character_pipe["prompt"],
            "system_instructions": pipe_system_instructions,
            "reference_images": normalized_reference_images,
            "character_plan_json": pipe_character_plan_json,
            "summary": pipe_summary,
        }

    def _resolve_generation_inputs(
        self,
        prompt,
        system_instructions="",
        reference_images=None,
        character_pipe=None,
    ):
        normalized_pipe = self._normalize_character_pipe(character_pipe)
        direct_prompt_present = self._is_non_empty_text(prompt)
        direct_system_present = self._is_non_empty_text(system_instructions)
        direct_reference_present = self._has_reference_images(reference_images)

        if normalized_pipe is not None:
            resolved_prompt = normalized_pipe["prompt"]
            if not self._is_non_empty_text(resolved_prompt):
                raise ValueError("Prompt is empty. Provide a prompt or connect character_pipe.")

            resolved_system_instructions = normalized_pipe["system_instructions"]
            pipe_reference_images = normalized_pipe["reference_images"]
            if self._has_reference_images(pipe_reference_images):
                resolved_reference_images = pipe_reference_images
            else:
                resolved_reference_images = None

            ignored_direct_inputs = (
                direct_prompt_present
                or direct_system_present
                or direct_reference_present
            )
        else:
            resolved_prompt = prompt
            if not self._is_non_empty_text(resolved_prompt):
                raise ValueError("Prompt is empty. Provide a prompt or connect character_pipe.")

            if direct_system_present:
                resolved_system_instructions = system_instructions
            else:
                resolved_system_instructions = ""

            if direct_reference_present:
                resolved_reference_images = reference_images
            else:
                resolved_reference_images = None

            ignored_direct_inputs = False

        return {
            "prompt": resolved_prompt,
            "system_instructions": resolved_system_instructions,
            "reference_images": resolved_reference_images,
            "character_plan_json": normalized_pipe["character_plan_json"] if normalized_pipe is not None else "",
            "planner_summary": normalized_pipe["summary"] if normalized_pipe is not None else "",
            "character_pipe_connected": normalized_pipe is not None,
            "ignored_direct_inputs": ignored_direct_inputs,
        }

    def _append_message_block(self, system_messages, block_text):
        if not self._is_non_empty_text(block_text):
            return system_messages

        if self._is_non_empty_text(system_messages):
            return f"{system_messages}\n\n{block_text}"
        return block_text

    def _append_character_pipe_override_note(self, system_messages, ignored_direct_inputs):
        if not ignored_direct_inputs:
            return system_messages

        return self._append_message_block(system_messages, CHARACTER_PIPE_OVERRIDE_NOTE)

    def _append_planner_summary(self, system_messages, planner_summary, character_pipe_connected):
        if not character_pipe_connected or not self._is_non_empty_text(planner_summary):
            return system_messages

        planner_block = f"Planner summary:\n{planner_summary}"
        return self._append_message_block(system_messages, planner_block)

    def _stringify_response_detail(self, value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()

        name = getattr(value, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()

        text = str(value).strip()
        if text in {"", "None"}:
            return ""
        return text

    def _collect_response_diagnostics(self, response, candidate):
        diagnostics = []

        finish_reason = self._stringify_response_detail(
            getattr(candidate, "finish_reason", None) if candidate is not None else None
        )
        if finish_reason:
            diagnostics.append(f"Finish reason: {finish_reason}")

        candidate_feedback = self._stringify_response_detail(
            getattr(candidate, "safety_ratings", None) if candidate is not None else None
        )
        if candidate_feedback:
            diagnostics.append(f"Candidate feedback: {candidate_feedback}")

        prompt_feedback = self._stringify_response_detail(getattr(response, "prompt_feedback", None))
        if prompt_feedback:
            diagnostics.append(f"Prompt feedback: {prompt_feedback}")

        return diagnostics

    def generate_image(
        self,
        prompt,
        model,
        resolution,
        aspect_ratio,
        seed,
        search_mode="legacy_toggles",
        enable_google_search=False,
        enable_image_search=False,
        thinking_mode="legacy_toggle",
        enable_thinking_mode=True,
        system_instructions="",
        reference_images=None,
        character_pipe=None,
    ):
        try:
            resolved_inputs = self._resolve_generation_inputs(
                prompt=prompt,
                system_instructions=system_instructions,
                reference_images=reference_images,
                character_pipe=character_pipe,
            )
        except ValueError as e:
            return self._blank_result(str(e))

        prompt = resolved_inputs["prompt"]
        system_instructions = resolved_inputs["system_instructions"]
        reference_images = resolved_inputs["reference_images"]
        planner_summary = resolved_inputs["planner_summary"]
        character_pipe_connected = resolved_inputs["character_pipe_connected"]
        ignored_direct_inputs = resolved_inputs["ignored_direct_inputs"]

        auth_error = self._get_provider_auth_error()
        if auth_error is not None:
            return self._blank_result(
                self._append_planner_summary(
                    self._append_character_pipe_override_note(auth_error, ignored_direct_inputs),
                    planner_summary,
                    character_pipe_connected,
                )
            )

        try:
            client = self._build_client()
            preflight_messages = []

            extra_flash31_aspects = {"1:4", "4:1", "1:8", "8:1"}

            requested_web_search, requested_image_search = self._resolve_search_mode(
                search_mode,
                enable_google_search,
                enable_image_search,
            )
            flash31_thinking_level, flash31_include_thoughts = self._resolve_thinking_mode(
                thinking_mode,
                enable_thinking_mode,
            )

            legacy_web_search = bool(enable_google_search)
            legacy_image_search = bool(enable_image_search)
            if search_mode != "legacy_toggles":
                if requested_web_search != legacy_web_search or requested_image_search != legacy_image_search:
                    preflight_messages.append(
                        "search_mode overrides legacy search toggles."
                    )

            legacy_thinking_level, legacy_include_thoughts = self._resolve_thinking_mode(
                "legacy_toggle",
                enable_thinking_mode,
            )
            if thinking_mode != "legacy_toggle":
                if (
                    flash31_thinking_level != legacy_thinking_level
                    or flash31_include_thoughts != legacy_include_thoughts
                ):
                    preflight_messages.append(
                        "thinking_mode overrides enable_thinking_mode toggle."
                    )

            web_search_enabled = requested_web_search
            image_search_enabled = requested_image_search

            selected_resolution = resolution
            selected_aspect_ratio = aspect_ratio

            if model in {"gemini-2.5-flash-image", "gemini-3-pro-image-preview"} and aspect_ratio in extra_flash31_aspects:
                selected_aspect_ratio = "1:1"
                preflight_messages.append(
                    f"Aspect ratio {aspect_ratio} is only supported by gemini-3.1-flash-image-preview. "
                    "Falling back to 1:1."
                )

            if model == "gemini-3-pro-image-preview" and resolution == "0.5K":
                selected_resolution = "1K"
                preflight_messages.append(
                    "Resolution 0.5K is only supported by gemini-3.1-flash-image-preview. Falling back to 1K."
                )

            if model == "gemini-2.5-flash-image" and (web_search_enabled or image_search_enabled):
                preflight_messages.append(
                    "Search tools are not used for gemini-2.5-flash-image in this node. Ignoring search settings."
                )
                web_search_enabled = False
                image_search_enabled = False

            if model == "gemini-3-pro-image-preview" and image_search_enabled:
                preflight_messages.append(
                    "Image search is only supported by gemini-3.1-flash-image-preview. Ignoring image search setting."
                )
                image_search_enabled = False

            if model != "gemini-3.1-flash-image-preview":
                if thinking_mode != "legacy_toggle":
                    preflight_messages.append(
                        "thinking_mode dropdown applies only to gemini-3.1-flash-image-preview in this node."
                    )
                elif not enable_thinking_mode:
                    preflight_messages.append(
                        "enable_thinking_mode toggle is only used by gemini-3.1-flash-image-preview in this node."
                    )

            contents = [prompt]

            # --- HANDLE REFERENCES AS A LIST ---
            if reference_images is not None:
                for img_tensor in reference_images:
                    if img_tensor is None:
                        continue
                    if not isinstance(img_tensor, torch.Tensor):
                        continue

                    # respect Gemini limit of 14:
                    if len(contents) - 1 >= 14:  # -1 for the text prompt
                        break

                    b = img_tensor.shape[0]
                    for i in range(b):
                        if len(contents) - 1 >= 14:
                            break
                        frame = img_tensor[i]
                        img_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np)
                        contents.append(pil_img)

            # Configure tools
            tools = []
            if model == "gemini-3.1-flash-image-preview":
                search_tools_enabled = web_search_enabled or image_search_enabled
                if search_tools_enabled:
                    has_advanced_search_types = all(
                        hasattr(types, name)
                        for name in ("SearchTypes", "WebSearch", "ImageSearch")
                    )

                    if has_advanced_search_types:
                        search_type_args = {}
                        if web_search_enabled:
                            search_type_args["web_search"] = types.WebSearch()
                        if image_search_enabled:
                            search_type_args["image_search"] = types.ImageSearch()

                        google_search_cfg = types.GoogleSearch(
                            search_types=types.SearchTypes(**search_type_args)
                        )
                        tools.append(types.Tool(google_search=google_search_cfg))
                    else:
                        if image_search_enabled and not web_search_enabled:
                            preflight_messages.append(
                                "Current google-genai version does not expose ImageSearch/SearchTypes. "
                                "Please upgrade google-genai to use image search grounding."
                            )
                        else:
                            tools.append(types.Tool(google_search=types.GoogleSearch()))
                            if image_search_enabled:
                                preflight_messages.append(
                                    "Current google-genai version does not expose ImageSearch/SearchTypes. "
                                    "Using web search only."
                                )
            elif model == "gemini-3-pro-image-preview" and web_search_enabled:
                tools.append(types.Tool(google_search=types.GoogleSearch()))

            config = None

            if model == "gemini-2.5-flash-image":
                # Request image output explicitly while keeping the config otherwise minimal.
                image_cfg = types.ImageConfig(
                    aspect_ratio=selected_aspect_ratio,
                )
                config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=image_cfg,
                    system_instruction=system_instructions or None,
                    seed=seed % 2147483647 if seed is not None else None,
                )
            elif model == "gemini-3.1-flash-image-preview":
                # Full config: expanded aspect ratios, optional image search, thinking toggle
                image_cfg = types.ImageConfig(
                    aspect_ratio=selected_aspect_ratio,
                    image_size=selected_resolution,
                )
                thinking_cfg = types.ThinkingConfig(
                    thinking_level=flash31_thinking_level,
                    include_thoughts=flash31_include_thoughts,
                )
                config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=image_cfg,
                    tools=tools if tools else None,
                    system_instruction=system_instructions or None,
                    seed=seed % 2147483647 if seed is not None else None,
                    thinking_config=thinking_cfg,
                )
            elif model == "gemini-3-pro-image-preview":
                # Full config: aspect ratio, resolution, thinking, text+image
                image_cfg = types.ImageConfig(
                    aspect_ratio=selected_aspect_ratio,
                    image_size=selected_resolution,
                )
                thinking_cfg = types.ThinkingConfig(
                    include_thoughts=True,
                    # optional: thinking_budget / thinking_level
                )
                config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=image_cfg,
                    tools=tools if tools else None,
                    system_instruction=system_instructions or None,
                    seed=seed % 2147483647 if seed is not None else None,
                    thinking_config=thinking_cfg,
                )
            else:
                raise ValueError(f"Unsupported model: {model}")

            # Call API
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            normal_images = []    # non-thinking images
            thinking_images = []  # thinking images (part.thought == True)
            thought_chunks = []
            answer_chunks = []
            system_messages = "\n".join(preflight_messages).strip()

            # Prefer the canonical structure from docs
            candidates = getattr(response, "candidates", None)
            parts = []
            first_candidate = None

            if candidates:
                # Be defensive: candidates may be empty or content/parts may be None
                first_candidate = candidates[0]
                content = getattr(first_candidate, "content", None)
                if content is not None:
                    parts = getattr(content, "parts", None) or []
            else:
                # response.parts might exist but be None, so use "or []"
                parts = getattr(response, "parts", None) or []

            # de-dupe using raw bytes so thought + final identical pair collapses to one
            seen_hashes = set()

            for part in parts:
                is_thought = getattr(part, "thought", False)

                # ----- image data -----
                if getattr(part, "inline_data", None):
                    image_bytes = part.inline_data.data

                    img_hash = hashlib.sha256(image_bytes).hexdigest()
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_np = np.array(pil_image).astype(np.float32) / 255.0
                    if image_np.ndim == 2:
                        # grayscale → expand to 3 channels
                        image_np = np.stack([image_np] * 3, axis=-1)

                    image_tensor = torch.from_numpy(image_np).unsqueeze(0)

                    if is_thought:
                        thinking_images.append(image_tensor)
                    else:
                        normal_images.append(image_tensor)

                # ----- text data -----
                txt = getattr(part, "text", None)
                if not txt:
                    continue

                if is_thought:
                    thought_chunks.append(txt)
                else:
                    answer_chunks.append(txt)

            # If there are no normal images but there ARE thinking images,
            # treat the first thinking image as a normal output as well
            if not normal_images and thinking_images:
                normal_images.append(thinking_images[0])

            # Build image batches; ALWAYS return at least 1 image per output
            if normal_images:
                normal_image_batch = torch.cat(normal_images, dim=0)
            else:
                normal_image_batch = torch.zeros((1, 64, 64, 3))
                no_image_messages = [
                    "No non-thinking image generated.",
                    "The request completed, but no usable image part was found in the response.",
                ]

            if thinking_images:
                thinking_image_batch = torch.cat(thinking_images, dim=0)
            else:
                # Return a blank placeholder instead of an empty batch,
                # so SaveImage doesn't crash if connected to this output.
                thinking_image_batch = torch.zeros((1, 64, 64, 3))

            thinking_process = "\n".join(thought_chunks).strip()
            answer_text = "\n".join(answer_chunks).strip()

            # For 3 Pro and 3.1 Flash Image we expect thought summaries when thinking output is enabled
            expects_thought_summary = (
                model == "gemini-3-pro-image-preview"
                or (model == "gemini-3.1-flash-image-preview" and flash31_include_thoughts)
            )

            if expects_thought_summary:
                if not thinking_process:
                    thinking_process = "No thought summary returned by the model."
            else:
                if not thinking_process:
                    thinking_process = ""

            if answer_text:
                if system_messages:
                    system_messages += "\n" + answer_text
                else:
                    system_messages = answer_text

            if not normal_images:
                if answer_text:
                    no_image_messages.append("Text response was returned without an image.")
                no_image_messages.extend(self._collect_response_diagnostics(response, first_candidate))
                no_image_block = "\n".join(no_image_messages)
                if system_messages:
                    system_messages += "\n" + no_image_block
                else:
                    system_messages = no_image_block

            system_messages = self._append_character_pipe_override_note(
                system_messages,
                ignored_direct_inputs,
            )
            system_messages = self._append_planner_summary(
                system_messages,
                planner_summary,
                character_pipe_connected,
            )

            return (
                normal_image_batch,
                thinking_image_batch,
                thinking_process,
                system_messages,
            )

        except Exception as e:
            return self._blank_result(
                self._append_planner_summary(
                    self._append_character_pipe_override_note(
                        f"Error: {str(e)}",
                        ignored_direct_inputs,
                    ),
                    planner_summary,
                    character_pipe_connected,
                )
            )


class BurveGoogleImageGen(BurveGoogleImageGenBase):
    PROVIDER_ID = "aistudio"

    def _get_provider_auth_error(self):
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            return None

        return (
            "Gemini API key not found.\n\n"
            "This node reads your key from the GEMINI_API_KEY environment variable.\n\n"
            "Set it like this and then restart ComfyUI:\n\n"
            "Windows (PowerShell):\n"
            "  setx GEMINI_API_KEY \"YOUR_REAL_KEY_HERE\"\n"
            "  # Then close and reopen your terminal / ComfyUI\n\n"
            "macOS / Linux (bash / zsh):\n"
            "  export GEMINI_API_KEY=\"YOUR_REAL_KEY_HERE\"\n"
            "  # If you want it permanent, add that line to ~/.bashrc or ~/.zshrc\n\n"
            "After setting GEMINI_API_KEY and restarting ComfyUI, run this node again."
        )

    def _build_client(self):
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())


class BurveVertexImageGen(BurveGoogleImageGenBase):
    PROVIDER_ID = "vertex"

    def _get_vertex_project(self):
        return os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()

    def _get_vertex_location(self):
        return os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()

    def _get_provider_auth_error(self):
        project = self._get_vertex_project()
        location = self._get_vertex_location()
        missing = []
        if not project:
            missing.append("GOOGLE_CLOUD_PROJECT")
        if not location:
            missing.append("GOOGLE_CLOUD_LOCATION")

        if not missing:
            return None

        missing_text = ", ".join(missing)
        return (
            "Vertex AI configuration is incomplete.\n\n"
            "This node ignores GEMINI_API_KEY.\n"
            "GOOGLE_APPLICATION_CREDENTIALS or ADC authenticates you, but it does not replace "
            "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.\n"
            "Credentials alone do not supply the required project and region for this node.\n\n"
            f"Missing required environment variables: {missing_text}\n\n"
            "Example setup:\n"
            "  export GOOGLE_CLOUD_PROJECT=\"your-gcp-project-id\"\n"
            "  export GOOGLE_CLOUD_LOCATION=\"us-central1\"\n\n"
            "Authentication:\n"
            "  - Set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON file, or\n"
            "  - Run: gcloud auth application-default login\n\n"
            "After setting the environment and restarting ComfyUI, run this node again."
        )

    def _build_client(self):
        return genai.Client(
            vertexai=True,
            project=self._get_vertex_project(),
            location=self._get_vertex_location(),
        )


class BurveImageRefPack:
    @classmethod
    def INPUT_TYPES(s):
        opt = {}
        # 14 optional IMAGE inputs
        for i in range(1, 15):
            opt[f"image{i}"] = ("IMAGE",)

        return {
            "required": {},
            "optional": opt,
        }

    # One output socket, custom type name
    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("images",)
    FUNCTION = "pack"
    CATEGORY = "BurveTools"

    def pack(self, **kwargs):
        """
        kwargs contains image1..image14, each either:
        - None, or
        - tensor [N, H, W, C]
        We return a Python list of tensors. Each item in the list can have its own size.
        """
        result = []

        for key, img in kwargs.items():
            if img is None:
                continue

            if isinstance(img, torch.Tensor):
                # keep batches as-is; Gemini node can iterate frames
                result.append(img)
            else:
                # Just in case someone passes something weird
                raise ValueError(f"{key} is not a tensor: {type(img)}")

        return (result,)


class BurveCharacterPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gender": (
                    ["female", "male"],
                    {"default": "female"},
                ),
                "age_years": ("INT", {"default": 25, "min": 18, "max": 80}),
                "race": (
                    RACE_OPTIONS,
                    {"default": "human"},
                ),
                "custom_race": ("STRING", {"multiline": False, "default": ""}),
                "height_cm": ("INT", {"default": 170, "min": 140, "max": 210}),
                "weight_kg": ("INT", {"default": 58, "min": 35, "max": 140}),
                "bust_cm": ("INT", {"default": 90, "min": 70, "max": 150}),
                "underbust_cm": ("INT", {"default": 74, "min": 55, "max": 130}),
                "waist_cm": ("INT", {"default": 64, "min": 45, "max": 130}),
                "full_hip_cm": ("INT", {"default": 95, "min": 70, "max": 170}),
                "male_chest_cm": ("INT", {"default": 102, "min": 80, "max": 160}),
                "body_frame_preset": (
                    ["balanced", "hourglass", "pear", "athletic"],
                    {"default": "balanced"},
                ),
                "male_body_frame_preset": (
                    ["balanced", "v_taper", "rectangular", "athletic", "stocky"],
                    {"default": "balanced"},
                ),
                "skin_tone": (
                    ["very_light", "light", "light_medium", "medium", "tan", "deep"],
                    {"default": "light_medium"},
                ),
                "custom_skin_tone": ("STRING", {"multiline": False, "default": ""}),
                "undertone": (
                    ["cool", "neutral", "warm", "olive"],
                    {"default": "neutral"},
                ),
                "hair_color": (
                    [
                        "dark_blonde",
                        "blonde",
                        "light_brown",
                        "brown",
                        "dark_brown",
                        "black",
                        "auburn",
                        "red",
                    ],
                    {"default": "dark_blonde"},
                ),
                "custom_hair_color": ("STRING", {"multiline": False, "default": ""}),
                "hair_length": (
                    ["bob", "shoulder_length", "long", "very_long"],
                    {"default": "long"},
                ),
                "musculature_tone": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "body_fat": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose": (
                    ["neutral_a_pose", "neutral_straight", "contrapposto_soft", "editorial_relaxed"],
                    {"default": "neutral_a_pose"},
                ),
                "outfit_variant": (
                    ["classic_triangle", "soft_scoop", "narrow_triangle", "halter_contour"],
                    {"default": "classic_triangle"},
                ),
                "male_outfit_variant": (
                    ["classic_brief", "square_cut", "short_boxer", "compression_short"],
                    {"default": "classic_brief"},
                ),
                "outfit_color": (
                    ["neutral_gray", "soft_taupe", "muted_black"],
                    {"default": "neutral_gray"},
                ),
                "use_face_reference": ("BOOLEAN", {"default": False}),
                "face_reference_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "race_override_pipe": ("CHARACTER_RACE_PIPE",),
                "face_reference_image": ("IMAGE",),
                "extra_reference_images": ("IMAGE_LIST",),
                "plan_overrides_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE_LIST", "STRING", "STRING", "CHARACTER_GEN_PIPE")
    RETURN_NAMES = ("prompt", "system_instructions", "reference_images", "character_plan_json", "summary", "character_pipe")
    FUNCTION = "build"
    CATEGORY = "BurveTools/Character"

    @staticmethod
    def _collect_ui_values(kwargs):
        return {
            "gender": kwargs["gender"],
            "age_years": kwargs["age_years"],
            "race": kwargs["race"],
            "custom_race": kwargs["custom_race"],
            "height_cm": kwargs["height_cm"],
            "weight_kg": kwargs["weight_kg"],
            "bust_cm": kwargs["bust_cm"],
            "underbust_cm": kwargs["underbust_cm"],
            "waist_cm": kwargs["waist_cm"],
            "full_hip_cm": kwargs["full_hip_cm"],
            "male_chest_cm": kwargs["male_chest_cm"],
            "body_frame_preset": kwargs["body_frame_preset"],
            "male_body_frame_preset": kwargs["male_body_frame_preset"],
            "skin_tone": kwargs["skin_tone"],
            "custom_skin_tone": kwargs["custom_skin_tone"],
            "undertone": kwargs["undertone"],
            "hair_color": kwargs["hair_color"],
            "custom_hair_color": kwargs["custom_hair_color"],
            "hair_length": kwargs["hair_length"],
            "musculature_tone": kwargs["musculature_tone"],
            "body_fat": kwargs["body_fat"],
            "pose": kwargs["pose"],
            "outfit_variant": kwargs["outfit_variant"],
            "male_outfit_variant": kwargs["male_outfit_variant"],
            "outfit_color": kwargs["outfit_color"],
            "use_face_reference": kwargs["use_face_reference"],
            "face_reference_strength": kwargs["face_reference_strength"],
        }

    @staticmethod
    def _normalize_input_types(input_types):
        if isinstance(input_types, list):
            if input_types:
                return input_types[0] or {}
            return {}
        return input_types or {}

    @classmethod
    def _input_is_connected(cls, input_types, name):
        normalized = cls._normalize_input_types(input_types)
        return name in normalized

    @staticmethod
    def _face_reference_present(face_reference_image):
        return (
            isinstance(face_reference_image, torch.Tensor)
            and face_reference_image.ndim >= 4
            and face_reference_image.shape[0] > 0
        )

    @staticmethod
    def _extra_reference_batch_sizes(extra_reference_images):
        batch_sizes = []
        if extra_reference_images is None:
            return batch_sizes

        for index, img_tensor in enumerate(extra_reference_images):
            if img_tensor is None:
                continue
            if not isinstance(img_tensor, torch.Tensor):
                raise ValueError(f"extra_reference_images[{index}] is not a tensor: {type(img_tensor)}")
            batch_sizes.append(int(img_tensor.shape[0]))

        return batch_sizes

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        ui_values = cls._collect_ui_values(kwargs)
        face_reference_present = cls._input_is_connected(input_types, "face_reference_image")

        try:
            build_character_plan(
                ui_values=ui_values,
                plan_overrides_json=kwargs.get("plan_overrides_json", ""),
                face_reference_present=face_reference_present,
                extra_reference_batch_sizes=[],
                race_override_pipe=kwargs.get("race_override_pipe"),
            )
        except ValueError as e:
            return str(e)

        return True

    def _pack_reference_images(self, face_reference_image, extra_reference_images, use_face_reference):
        packed = []

        if use_face_reference and self._face_reference_present(face_reference_image):
            packed.append(face_reference_image[:1])

        if extra_reference_images is None:
            return packed

        for index, img_tensor in enumerate(extra_reference_images):
            if img_tensor is None:
                continue
            if not isinstance(img_tensor, torch.Tensor):
                raise ValueError(f"extra_reference_images[{index}] is not a tensor: {type(img_tensor)}")

            batch_size = img_tensor.shape[0]
            for frame_index in range(batch_size):
                if len(packed) >= 14:
                    return packed
                packed.append(img_tensor[frame_index:frame_index + 1])

        return packed

    @staticmethod
    def _build_character_pipe(result, reference_images, summary):
        return {
            "kind": CHARACTER_GEN_PIPE_KIND,
            "version": CHARACTER_GEN_PIPE_VERSION,
            "prompt": result["prompt"],
            "system_instructions": result["system_instructions"],
            "reference_images": reference_images,
            "character_plan_json": result["character_plan_json"],
            "summary": summary,
        }

    def build(
        self,
        gender,
        age_years,
        race,
        custom_race,
        height_cm,
        weight_kg,
        bust_cm,
        underbust_cm,
        waist_cm,
        full_hip_cm,
        male_chest_cm,
        body_frame_preset,
        male_body_frame_preset,
        skin_tone,
        custom_skin_tone,
        undertone,
        hair_color,
        custom_hair_color,
        hair_length,
        musculature_tone,
        body_fat,
        pose,
        outfit_variant,
        male_outfit_variant,
        outfit_color,
        use_face_reference,
        face_reference_strength,
        race_override_pipe=None,
        face_reference_image=None,
        extra_reference_images=None,
        plan_overrides_json="",
    ):
        ui_values = self._collect_ui_values(
            {
                "gender": gender,
                "age_years": age_years,
                "race": race,
                "custom_race": custom_race,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "bust_cm": bust_cm,
                "underbust_cm": underbust_cm,
                "waist_cm": waist_cm,
                "full_hip_cm": full_hip_cm,
                "male_chest_cm": male_chest_cm,
                "body_frame_preset": body_frame_preset,
                "male_body_frame_preset": male_body_frame_preset,
                "skin_tone": skin_tone,
                "custom_skin_tone": custom_skin_tone,
                "undertone": undertone,
                "hair_color": hair_color,
                "custom_hair_color": custom_hair_color,
                "hair_length": hair_length,
                "musculature_tone": musculature_tone,
                "body_fat": body_fat,
                "pose": pose,
                "outfit_variant": outfit_variant,
                "male_outfit_variant": male_outfit_variant,
                "outfit_color": outfit_color,
                "use_face_reference": use_face_reference,
                "face_reference_strength": face_reference_strength,
            }
        )

        face_reference_present = self._face_reference_present(face_reference_image)
        extra_batch_sizes = self._extra_reference_batch_sizes(extra_reference_images)
        result = build_character_plan(
            ui_values=ui_values,
            plan_overrides_json=plan_overrides_json,
            face_reference_present=face_reference_present,
            extra_reference_batch_sizes=extra_batch_sizes,
            race_override_pipe=race_override_pipe,
        )

        reference_images = self._pack_reference_images(
            face_reference_image=face_reference_image,
            extra_reference_images=extra_reference_images,
            use_face_reference=result["plan"]["identity"]["face_reference"]["enabled"],
        )

        summary = result["summary"]
        if (
            use_face_reference
            and isinstance(face_reference_image, torch.Tensor)
            and face_reference_image.ndim >= 4
            and face_reference_image.shape[0] > 1
        ):
            summary += (
                "\nNode note: face_reference_image contained multiple frames; only the first frame was used "
                "as the dedicated face anchor."
            )

        character_pipe = self._build_character_pipe(
            result=result,
            reference_images=reference_images,
            summary=summary,
        )

        return (
            result["prompt"],
            result["system_instructions"],
            reference_images,
            result["character_plan_json"],
            summary,
            character_pipe,
        )


class BurveCharacterRaceDetails:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "race_name": (
                    RACE_NAME_OPTIONS,
                    {"default": "none"},
                ),
                "custom_race_name": ("STRING", {"multiline": False, "default": ""}),
                "ears": (RACE_DETAIL_OPTIONS["ears"], {"default": "none"}),
                "custom_ears": ("STRING", {"multiline": False, "default": ""}),
                "horns": (RACE_DETAIL_OPTIONS["horns"], {"default": "none"}),
                "custom_horns": ("STRING", {"multiline": False, "default": ""}),
                "wings": (RACE_DETAIL_OPTIONS["wings"], {"default": "none"}),
                "custom_wings": ("STRING", {"multiline": False, "default": ""}),
                "tail": (RACE_DETAIL_OPTIONS["tail"], {"default": "none"}),
                "custom_tail": ("STRING", {"multiline": False, "default": ""}),
                "legs_feet": (RACE_DETAIL_OPTIONS["legs_feet"], {"default": "none"}),
                "custom_legs_feet": ("STRING", {"multiline": False, "default": ""}),
                "skin_surface": (RACE_DETAIL_OPTIONS["skin_surface"], {"default": "none"}),
                "custom_skin_surface": ("STRING", {"multiline": False, "default": ""}),
                "head_features": (RACE_DETAIL_OPTIONS["head_features"], {"default": "none"}),
                "custom_head_features": ("STRING", {"multiline": False, "default": ""}),
                "hands_arms": (RACE_DETAIL_OPTIONS["hands_arms"], {"default": "none"}),
                "custom_hands_arms": ("STRING", {"multiline": False, "default": ""}),
                "extra_notes": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("CHARACTER_RACE_PIPE", "STRING", "STRING")
    RETURN_NAMES = ("race_override_pipe", "race_override_json", "summary")
    FUNCTION = "build"
    CATEGORY = "BurveTools/Character"

    @staticmethod
    def _resolve_text_or_selection(selection, custom_text):
        custom_value = str(custom_text).strip()
        if custom_value:
            return custom_value, True
        if selection != "none":
            return selection, False
        return "", False

    def build(
        self,
        race_name,
        custom_race_name,
        ears,
        custom_ears,
        horns,
        custom_horns,
        wings,
        custom_wings,
        tail,
        custom_tail,
        legs_feet,
        custom_legs_feet,
        skin_surface,
        custom_skin_surface,
        head_features,
        custom_head_features,
        hands_arms,
        custom_hands_arms,
        extra_notes,
    ):
        resolved_race_name, custom_race_used = self._resolve_text_or_selection(race_name, custom_race_name)
        trait_inputs = {
            "ears": (ears, custom_ears),
            "horns": (horns, custom_horns),
            "wings": (wings, custom_wings),
            "tail": (tail, custom_tail),
            "legs_feet": (legs_feet, custom_legs_feet),
            "skin_surface": (skin_surface, custom_skin_surface),
            "head_features": (head_features, custom_head_features),
            "hands_arms": (hands_arms, custom_hands_arms),
        }

        traits = {}
        custom_overrides = []
        active_traits = []
        for key in RACE_TRAIT_KEYS:
            if key == "extra_notes":
                value = str(extra_notes).strip()
                traits[key] = value
                if value:
                    custom_overrides.append("extra_notes")
                    active_traits.append("extra_notes")
                continue

            selected_value, custom_used = self._resolve_text_or_selection(*trait_inputs[key])
            traits[key] = selected_value
            if custom_used:
                custom_overrides.append(key)
            if selected_value:
                active_traits.append(f"{key}={selected_value}")

        if custom_race_used:
            custom_overrides.insert(0, "race_name")

        race_override_pipe = {
            "kind": CHARACTER_RACE_PIPE_KIND,
            "version": CHARACTER_RACE_PIPE_VERSION,
            "race_name": resolved_race_name,
            "traits": traits,
            "summary": "",
        }

        summary_lines = [
            f"Race name: {resolved_race_name or 'none'}",
            f"Active traits: {', '.join(active_traits) if active_traits else 'none'}",
            f"Custom text overrides: {', '.join(custom_overrides) if custom_overrides else 'none'}",
        ]
        summary = "\n".join(summary_lines)
        race_override_pipe["summary"] = summary
        race_override_json = json.dumps(race_override_pipe, indent=2, ensure_ascii=True)
        return (race_override_pipe, race_override_json, summary)

class BurveDebugGeminiKey:
    @classmethod
    def INPUT_TYPES(cls):
        # No inputs – just inspects the environment
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    CATEGORY = "BurveTools/Debug"
    OUTPUT_NODE = True  # can be selected as a workflow output

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force this node to be treated as "changed" every run
        # so Comfy doesn't cache-skip it
        return float("NaN")

    def run(self):
        key = os.getenv("GEMINI_API_KEY", "").strip()

        if not key:
            info = (
                "GEMINI_API_KEY is NOT set or is empty inside ComfyUI.\n\n"
                "Set it in your environment and restart Comfy."
            )
        else:
            info = (
                "GEMINI_API_KEY detected inside ComfyUI:\n"
                f"  length = {len(key)}\n"
                f"  start  = {key[:4]}\n"
                f"  end    = {key[-4:]}\n\n"
                "Full key is NOT shown for safety."
            )

        # Print to terminal
        print("========== [BurveDebugGeminiKey] ==========")
        print(info)
        print("===========================================")

        # Also return it so UI nodes can display it
        return (info,)


class BurveDebugVertexAuth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    CATEGORY = "BurveTools/Debug"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def run(self):
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        auth_status = "yes" if credentials_path else "maybe"
        missing_config = []
        if not project:
            missing_config.append("GOOGLE_CLOUD_PROJECT")
        if not location:
            missing_config.append("GOOGLE_CLOUD_LOCATION")

        info_lines = [
            "Vertex AI environment status inside ComfyUI:",
            f"  GOOGLE_CLOUD_PROJECT present: {'yes' if project else 'no'}",
            f"  GOOGLE_CLOUD_LOCATION present: {'yes' if location else 'no'}",
            f"  GOOGLE_APPLICATION_CREDENTIALS present: {'yes' if credentials_path else 'no'}",
            f"  Authentication configured: {auth_status}",
            "",
            "This node uses standard Vertex AI auth, not GEMINI_API_KEY.",
            "Credentials authenticate the request.",
            "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are still required by this node.",
            "GOOGLE_APPLICATION_CREDENTIALS present: yes does not mean the node is fully configured.",
            "ADC may also come from: gcloud auth application-default login",
        ]

        if project:
            info_lines.append(f"  project = {project}")
        if location:
            info_lines.append(f"  location = {location}")
        if credentials_path:
            info_lines.append(f"  credentials_path = {credentials_path}")
        if missing_config:
            info_lines.append(f"Missing required config: {', '.join(missing_config)}")

        info = "\n".join(info_lines)

        print("========== [BurveDebugVertexAuth] ==========")
        print(info)
        print("============================================")

        return (info,)


class BurveSystemInstructions:
    @classmethod
    def INPUT_TYPES(s):
        data = load_instructions()
        presets = data.get("presets", [])
        # Create a list of titles for the dropdown
        # We'll map titles back to the full instruction in the run method
        # If no presets, provide a placeholder
        if not presets:
            names = ["No instructions found"]
        else:
            names = [p.get("title", "Unknown") for p in presets]
        
        return {
            "required": {
                "instruction_name": (names,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instruction",)
    FUNCTION = "get_instruction"
    CATEGORY = "BurveTools"

    def get_instruction(self, instruction_name):
        data = load_instructions()
        presets = data.get("presets", [])
        
        selected_instruction = ""
        for p in presets:
            if p.get("title") == instruction_name:
                selected_instruction = p.get("system", "")
                break
        
        return (selected_instruction,)

# --- Auto-update logic for prompts ---
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")
PROMPTS_URL = "https://raw.githubusercontent.com/Burve/ComfyUI_Burve_Tools/main/prompts.json"

def load_prompts():
    if not os.path.exists(PROMPTS_FILE):
        return {"version": 0, "prompts": []}
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[BurveTools] Error loading prompts: {e}")
        return {"version": 0, "prompts": []}

def update_prompts():
    print("[BurveTools] Checking for prompt updates...")
    current_data = load_prompts()
    current_version = current_data.get("version", 0)

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(PROMPTS_URL, context=ctx, timeout=5) as response:
            if response.status == 200:
                remote_data = json.loads(response.read().decode('utf-8'))
                remote_version = remote_data.get("version", 0)
                
                if remote_version > current_version:
                    print(f"[BurveTools] Updating prompts from v{current_version} to v{remote_version}")
                    with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(remote_data, f, indent=2)
                else:
                    print("[BurveTools] Prompts are up to date.")
    except Exception as e:
        print(f"[BurveTools] Prompt update check failed: {e}")

# Run update on module load
update_prompts()

class BurveVariableInjector:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {}
        for i in range(1, 15):
            inputs[f"V{i}"] = ("STRING", {"multiline": False, "default": ""})
        
        return {
            "required": {},
            "optional": inputs
        }

    RETURN_TYPES = ("VARIABLE_DICT",)
    RETURN_NAMES = ("variables",)
    FUNCTION = "inject_variables"
    CATEGORY = "BurveTools"

    def inject_variables(self, **kwargs):
        # Filter out empty strings if desired, or keep them. 
        # The user request implies we just pass them all.
        # We'll pass everything that is provided.
        return (kwargs,)

class BurvePromptDatabase:
    @classmethod
    def INPUT_TYPES(cls):
        data = load_prompts()
        prompts = data.get("prompts", [])

        if not prompts:
            names = ["No prompts found"]
        else:
            # generate labels like "[Category] Title"
            labels = []
            for p in prompts:
                title = p.get("title", "Unknown")
                category = p.get("category", "Uncategorized")
                label = f"[{category}] {title}"
                labels.append(label)

            # sort alphabetically (case-insensitive)
            names = sorted(labels, key=lambda x: x.lower())

        return {
            "required": {
                "prompt_name": (names,),
            },
            "optional": {
                # your variables dict input
                "variables": ("VARIABLE_DICT",),
            }
        }

    # now we have a 3rd output for the plain title
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("compiled_prompt", "raw_prompt", "title")
    FUNCTION = "process_prompt"
    CATEGORY = "BurveTools"

    def process_prompt(self, prompt_name, variables=None):
        data = load_prompts()
        prompts = data.get("prompts", [])

        raw_prompt = ""
        raw_title = ""

        for p in prompts:
            title = p.get("title", "Unknown")
            category = p.get("category", "Uncategorized")
            label = f"[{category}] {title}"

            # Compare against the label shown in the dropdown
            if label == prompt_name:
                # store the plain title for output
                raw_title = title
                # In your JSON the field is called "system"
                raw_prompt = p.get("system") or p.get("prompt", "")
                break

        if not raw_prompt:
            return ("", "", "")

        if variables is None:
            variables = {}

        import re
        # Regex to find [[variable_name:default_value]]
        pattern = r"\[\[([a-zA-Z_]\w*):([^\]]*)\]\]"

        def replace_match(match):
            var_name = match.group(1)
            default_val = match.group(2)

            user_value = variables.get(var_name)

            # If user_value is present and not empty, use it. Otherwise default.
            if user_value:
                return str(user_value)
            return default_val

        compiled_prompt = re.sub(pattern, replace_match, raw_prompt)

        # compiled, raw, and the plain title
        return (compiled_prompt, raw_prompt, raw_title)

class BurveBlindGridSplitter:
    """
    Blind grid splitter:
    - Input: IMAGE (B, H, W, C)
    - Params: rows, cols
    - Output: IMAGE batch containing all tiles (B * rows * cols, tile_h, tile_w, C)
    - No OpenCV, no analysis, just straight slicing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 64}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 64}),
                # If image size is not perfectly divisible, you can center-crop
                # the used area instead of from top-left.
                "center_crop": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tiles",)
    FUNCTION = "split"
    CATEGORY = "BurveTools"

    def split(self, image, rows, cols, center_crop=False):
        # image: torch tensor (B, H, W, C)
        b, h, w, c = image.shape

        # Compute tile size from full image size
        tile_h = h // rows
        tile_w = w // cols

        # If tiles would be smaller than 1 px, bail out safely.
        if tile_h < 1 or tile_w < 1:
            print(
                f"BurveBlindGridSplitter: image {h}x{w} too small for "
                f"{rows}x{cols} grid, returning original image."
            )
            return (image,)

        used_h = tile_h * rows
        used_w = tile_w * cols

        # Optionally center-crop the area we actually use
        if center_crop:
            top = max((h - used_h) // 2, 0)
            left = max((w - used_w) // 2, 0)
        else:
            top = 0
            left = 0

        bottom = top + used_h
        right = left + used_w

        cropped = image[:, top:bottom, left:right, :]

        tiles = []
        for bi in range(b):
            for r in range(rows):
                for col in range(cols):
                    y0 = r * tile_h
                    y1 = y0 + tile_h
                    x0 = col * tile_w
                    x1 = x0 + tile_w
                    # Keep batch dim = 1 for each tile so cat works
                    tile = cropped[bi:bi + 1, y0:y1, x0:x1, :]
                    tiles.append(tile)

        if not tiles:
            # Extremely defensive fallback; shouldn't happen.
            return (image,)

        out = torch.cat(tiles, dim=0)
        return (out,)


class BurvePromptSelector14:
    """
    Prompt Selector 14:
    - Stores up to 14 prompt strings
    - Selects one based on index (1–14)
    - Always errors if index is out of bounds (not 1-14)
    - Optional error when selected prompt is empty
    - Outputs: selected_prompt (STRING), selected_index (INT)
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        # 14 multiline prompt inputs
        for i in range(1, 15):
            required[f"prompt_{i}"] = ("STRING", {"multiline": True, "default": ""})

        required["index"] = ("INT", {"default": 1, "min": 1, "max": 14, "step": 1})
        required["error_on_empty"] = ("BOOLEAN", {"default": True})

        return {
            "required": required,
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("selected_prompt", "selected_index")
    FUNCTION = "select"
    CATEGORY = "BurveTools/Text"

    def _cancel_remaining_queue(self):
        """
        Best-effort cancellation of remaining queued runs.
        Wrapped in try/except to avoid crashing if APIs are unavailable.
        """
        # Attempt to interrupt current processing
        try:
            import comfy.model_management
            if hasattr(comfy.model_management, "interrupt_current_processing"):
                comfy.model_management.interrupt_current_processing()
        except Exception:
            pass

        # Attempt to wipe the prompt queue
        try:
            import server
            if hasattr(server, "PromptServer"):
                ps = server.PromptServer.instance
                if ps is not None and hasattr(ps, "prompt_queue"):
                    pq = ps.prompt_queue
                    if hasattr(pq, "wipe"):
                        pq.wipe()
        except Exception:
            pass

    def select(
        self,
        index,
        error_on_empty,
        **kwargs,
    ):
        # Build prompt dict from kwargs (prompt_1 through prompt_14)
        prompts = {}
        for i in range(1, 15):
            key = f"prompt_{i}"
            prompts[i] = kwargs.get(key, "")

        # Always error if index is out of bounds (not between 1 and 14)
        if index < 1 or index > 14:
            self._cancel_remaining_queue()
            raise ValueError(
                f"[BurvePromptSelector14] Index {index} is out of bounds! "
                f"Must be between 1 and 14."
            )

        selected_index = index

        # Get the selected prompt
        selected_prompt = prompts.get(selected_index, "")

        # Check for empty prompt if toggle is enabled
        if error_on_empty and (not selected_prompt or not selected_prompt.strip()):
            # Attempt to cancel remaining queue
            self._cancel_remaining_queue()

            raise ValueError(
                f"[BurvePromptSelector14] Prompt {selected_index} is empty! "
                f"Stopping execution. (error_on_empty=True)"
            )

        return (selected_prompt, selected_index)
