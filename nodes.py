import hashlib
import math
import json
import os
import random
import ssl
import time
import urllib.request
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
try:
    from PIL.PngImagePlugin import PngInfo
except ImportError:
    PngInfo = None
from google import genai
from google.genai import errors as genai_errors

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from .gemini_image_service import (
        BACKOFF_EXP_BASE,
        BACKOFF_INITIAL_DELAY_SECONDS,
        BACKOFF_JITTER_MAX,
        BACKOFF_JITTER_MIN,
        BACKOFF_MAX_DELAY_SECONDS,
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
        DEFAULT_RETRY_ATTEMPTS,
        RETRYABLE_STATUS_CODES,
        SYSTEM_MESSAGES_DEFAULT,
        THINKING_PROCESS_DEFAULT,
        append_message_block as append_service_message_block,
        blank_image_tensor,
        clone_config_with_http_timeout,
        ensure_default_text,
        GeminiAspectRatioCompatibilityError,
        parse_generate_content_response,
        prepare_generate_content_request,
    )
except ImportError:
    from gemini_image_service import (
        BACKOFF_EXP_BASE,
        BACKOFF_INITIAL_DELAY_SECONDS,
        BACKOFF_JITTER_MAX,
        BACKOFF_JITTER_MIN,
        BACKOFF_MAX_DELAY_SECONDS,
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
        DEFAULT_RETRY_ATTEMPTS,
        RETRYABLE_STATUS_CODES,
        SYSTEM_MESSAGES_DEFAULT,
        THINKING_PROCESS_DEFAULT,
        append_message_block as append_service_message_block,
        blank_image_tensor,
        clone_config_with_http_timeout,
        ensure_default_text,
        GeminiAspectRatioCompatibilityError,
        parse_generate_content_response,
        prepare_generate_content_request,
    )

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
GENERATED_IMAGE_PIPE_KIND = "burve.generated_image_pipe"
GENERATED_IMAGE_PIPE_VERSION = 1
CROP_REGION_PIPE_KIND = "burve.crop_region_pipe"
CROP_REGION_PIPE_VERSION = 1
CHARACTER_PIPE_OVERRIDE_NOTE = (
    "character_pipe is connected; ignoring direct prompt, system_instructions, and "
    "reference_images inputs."
)
BURVE_CROP_MASK_ASPECT_RATIOS = (
    "1:1",
    "3:2",
    "2:3",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
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


class GeminiRequestTimeoutError(RuntimeError):
    def __init__(self, timeout_seconds, attempts_made, last_error=None):
        super().__init__(f"Request timed out after {timeout_seconds}s.")
        self.timeout_seconds = timeout_seconds
        self.attempts_made = attempts_made
        self.last_error = last_error


class GeminiRequestRetryExhaustedError(RuntimeError):
    def __init__(self, attempts_made, last_error):
        super().__init__(f"Request failed after {attempts_made} attempts.")
        self.attempts_made = attempts_made
        self.last_error = last_error

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
        # Public node schema is defined in v3_extension.py. This fallback exists only
        # for direct Python use and keeps the common inputs visible without legacy fields.
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (
                    ["gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview", "gemini-2.5-flash-image"],
                    {"default": "gemini-2.5-flash-image"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "system_instructions": ("STRING", {"multiline": True, "default": ""}),
                "reference_images": ("IMAGE_LIST",),
                "character_pipe": ("CHARACTER_GEN_PIPE",),
                "aspect_ratio_override": ("STRING", {"multiline": False, "default": ""}),
                "request_timeout_seconds": (
                    "INT",
                    {
                        "default": DEFAULT_REQUEST_TIMEOUT_SECONDS,
                        "min": 10,
                        "max": 1800,
                        "advanced": True,
                    },
                ),
                "retry_attempts": (
                    "INT",
                    {
                        "default": DEFAULT_RETRY_ATTEMPTS,
                        "min": 1,
                        "max": 10,
                        "advanced": True,
                    },
                ),
            },
        }

    # normal images, thinking images, thinking text, system messages
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "GENERATED_IMAGE_PIPE")
    RETURN_NAMES = ("image", "thinking_image", "thinking_process", "system_messages", "generated_image_pipe")
    FUNCTION = "generate_image"
    CATEGORY = "BurveTools"

    def _get_provider_auth_error(self):
        raise NotImplementedError

    def _build_client(self):
        raise NotImplementedError

    def _blank_result(self, system_messages):
        blank = blank_image_tensor()
        return (
            blank,
            blank.clone(),
            THINKING_PROCESS_DEFAULT,
            ensure_default_text(system_messages, SYSTEM_MESSAGES_DEFAULT),
            self._empty_generated_image_pipe(model_id="", note="No image was generated."),
        )

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
        return append_service_message_block(system_messages, block_text)

    def _append_character_pipe_override_note(self, system_messages, ignored_direct_inputs):
        if not ignored_direct_inputs:
            return system_messages

        return self._append_message_block(system_messages, CHARACTER_PIPE_OVERRIDE_NOTE)

    def _append_planner_summary(self, system_messages, planner_summary, character_pipe_connected):
        if not character_pipe_connected or not self._is_non_empty_text(planner_summary):
            return system_messages

        planner_block = f"Planner summary:\n{planner_summary}"
        return self._append_message_block(system_messages, planner_block)

    def _mime_type_to_extension(self, mime_type):
        return {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
        }.get(mime_type)

    def _empty_generated_image_pipe(self, model_id, note=""):
        return {
            "kind": GENERATED_IMAGE_PIPE_KIND,
            "version": GENERATED_IMAGE_PIPE_VERSION,
            "provider_id": self.PROVIDER_ID,
            "model_id": model_id,
            "items": [],
            "notes": note or "",
        }

    def _build_generated_image_pipe(self, parsed_response, provider_id, model_id):
        artifacts = tuple(getattr(parsed_response, "image_artifacts", ()) or ())
        if not artifacts:
            return self._empty_generated_image_pipe(
                model_id=model_id,
                note="No normal image artifacts were available in the response.",
            )

        items = []
        notes = []
        passthrough_supported = provider_id == "vertex"
        if not passthrough_supported:
            notes.append("Exact passthrough bytes are unavailable on the Gemini API SDK path.")

        for artifact in artifacts:
            mime_type = getattr(artifact, "mime_type", None)
            extension = self._mime_type_to_extension(mime_type)
            raw_bytes = getattr(artifact, "raw_bytes", None)
            source = getattr(artifact, "source", "unavailable")
            if passthrough_supported and raw_bytes is not None and extension is not None:
                item_raw_bytes = raw_bytes
                item_source = source or "response_bytes"
            else:
                item_raw_bytes = None
                item_source = "unavailable"
            items.append(
                {
                    "mime_type": mime_type,
                    "extension": extension,
                    "raw_bytes": item_raw_bytes,
                    "sha256": getattr(artifact, "sha256", ""),
                    "source": item_source,
                }
            )

        return {
            "kind": GENERATED_IMAGE_PIPE_KIND,
            "version": GENERATED_IMAGE_PIPE_VERSION,
            "provider_id": provider_id,
            "model_id": model_id,
            "items": items,
            "notes": "\n".join(notes),
        }

    def _normalize_generated_image_pipe(self, generated_image_pipe):
        if generated_image_pipe is None:
            return None
        if not isinstance(generated_image_pipe, dict):
            raise ValueError("Invalid generated_image_pipe: expected a dict payload.")
        if generated_image_pipe.get("kind") != GENERATED_IMAGE_PIPE_KIND:
            raise ValueError("Invalid generated_image_pipe: unexpected kind.")
        if generated_image_pipe.get("version") != GENERATED_IMAGE_PIPE_VERSION:
            raise ValueError("Invalid generated_image_pipe: unsupported version.")
        items = generated_image_pipe.get("items", [])
        if not isinstance(items, list):
            raise ValueError("Invalid generated_image_pipe: items must be a list.")

        normalized_items = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid generated_image_pipe: item {index} must be a dict.")
            raw_bytes = item.get("raw_bytes")
            if raw_bytes is not None and not isinstance(raw_bytes, (bytes, bytearray)):
                raise ValueError(f"Invalid generated_image_pipe: item {index} raw_bytes must be bytes.")
            mime_type = item.get("mime_type")
            if mime_type is not None and not isinstance(mime_type, str):
                raise ValueError(f"Invalid generated_image_pipe: item {index} mime_type must be a string.")
            normalized_items.append(
                {
                    "mime_type": mime_type,
                    "extension": item.get("extension"),
                    "raw_bytes": bytes(raw_bytes) if raw_bytes is not None else None,
                    "sha256": str(item.get("sha256", "")),
                    "source": str(item.get("source", "unavailable")),
                }
            )

        notes = generated_image_pipe.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("Invalid generated_image_pipe: notes must be a string.")

        return {
            "kind": GENERATED_IMAGE_PIPE_KIND,
            "version": GENERATED_IMAGE_PIPE_VERSION,
            "provider_id": str(generated_image_pipe.get("provider_id", "")),
            "model_id": str(generated_image_pipe.get("model_id", "")),
            "items": normalized_items,
            "notes": notes,
        }

    def _normalize_request_timeout_seconds(self, value):
        if not isinstance(value, int):
            raise ValueError("request_timeout_seconds must be an integer.")
        if value < 10 or value > 1800:
            raise ValueError("request_timeout_seconds must be between 10 and 1800.")
        return value

    def _normalize_retry_attempts(self, value):
        if not isinstance(value, int):
            raise ValueError("retry_attempts must be an integer.")
        if value < 1 or value > 10:
            raise ValueError("retry_attempts must be between 1 and 10.")
        return value

    def _is_timeout_like_error(self, error):
        if isinstance(error, TimeoutError):
            return True

        class_name = type(error).__name__.lower()
        module_name = type(error).__module__.lower()
        message = str(error).lower()

        if "timeout" in class_name or "timed out" in message:
            return True

        timeout_modules = ("httpx", "httpcore", "socket")
        if module_name.startswith(timeout_modules) and "timeout" in class_name:
            return True

        return False

    def _is_connection_like_error(self, error):
        class_name = type(error).__name__.lower()
        module_name = type(error).__module__.lower()
        connection_markers = (
            "connecterror",
            "readerror",
            "remoteprotocolerror",
            "networkerror",
            "protocolerror",
        )
        if class_name in connection_markers:
            return True
        return module_name.startswith(("httpx", "httpcore")) and class_name in connection_markers

    def _is_retryable_generate_error(self, error):
        if isinstance(error, genai_errors.APIError):
            return getattr(error, "code", None) in RETRYABLE_STATUS_CODES
        return self._is_timeout_like_error(error) or self._is_connection_like_error(error)

    def _compute_backoff_delay_seconds(self, attempt_index):
        base_delay = min(
            BACKOFF_INITIAL_DELAY_SECONDS * (BACKOFF_EXP_BASE ** attempt_index),
            BACKOFF_MAX_DELAY_SECONDS,
        )
        return base_delay * random.uniform(BACKOFF_JITTER_MIN, BACKOFF_JITTER_MAX)

    def _request_generate_content(
        self,
        *,
        client,
        request,
        request_timeout_seconds,
        retry_attempts,
    ):
        deadline = time.monotonic() + request_timeout_seconds
        attempts_made = 0
        last_retryable_error = None

        for attempt_index in range(retry_attempts):
            attempts_made = attempt_index + 1
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise GeminiRequestTimeoutError(
                    request_timeout_seconds,
                    max(attempts_made - 1, 0),
                    last_retryable_error,
                )

            timed_config = clone_config_with_http_timeout(
                request.config,
                max(int(remaining_seconds * 1000), 1),
            )

            try:
                return client.models.generate_content(
                    model=request.model_id,
                    contents=request.contents,
                    config=timed_config,
                )
            except Exception as error:
                if not self._is_retryable_generate_error(error):
                    raise

                last_retryable_error = error
                if attempt_index >= retry_attempts - 1:
                    raise GeminiRequestRetryExhaustedError(attempts_made, error) from error

                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise GeminiRequestTimeoutError(
                        request_timeout_seconds,
                        attempts_made,
                        error,
                    ) from error

                delay_seconds = min(
                    self._compute_backoff_delay_seconds(attempt_index),
                    remaining_seconds,
                )
                if delay_seconds <= 0:
                    raise GeminiRequestTimeoutError(
                        request_timeout_seconds,
                        attempts_made,
                        error,
                    ) from error
                time.sleep(delay_seconds)

        raise GeminiRequestRetryExhaustedError(attempts_made, last_retryable_error)

    def _describe_retry_error(self, error):
        code = getattr(error, "code", None)
        status = getattr(error, "status", None)
        details = []
        if code is not None:
            details.append(f"status {code}")
        if status:
            details.append(str(status))
        if details:
            return ", ".join(details)
        return str(error).strip()

    def _format_generation_error(self, error, model_id, request_timeout_seconds):
        if isinstance(error, GeminiRequestTimeoutError):
            message = (
                f"Error: Request timed out after {request_timeout_seconds}s waiting for "
                "Gemini image generation."
            )
            if error.attempts_made:
                message += f"\nAttempts completed: {error.attempts_made}."
            if error.last_error is not None and str(error.last_error).strip():
                message += f"\nLast error: {error.last_error}"
        elif isinstance(error, GeminiRequestRetryExhaustedError):
            retry_detail = self._describe_retry_error(error.last_error)
            message = f"Error: Request failed after {error.attempts_made} attempts."
            if retry_detail:
                message += f"\nLast error: {retry_detail}"
        else:
            message = f"Error: {str(error)}"

        if model_id == "gemini-3.1-flash-image-preview":
            message += (
                "\nTry global location, lower resolution, MINIMAL thinking, or a "
                "higher advanced timeout."
            )
        return message

    def _resolve_model_id_for_error(self, model):
        if isinstance(model, dict):
            model_id = model.get("model")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
        if isinstance(model, str) and model.strip():
            return model.strip()
        return ""

    def generate_image(
        self,
        prompt,
        model,
        seed,
        system_instructions="",
        reference_images=None,
        character_pipe=None,
        aspect_ratio_override="",
        request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        retry_attempts=DEFAULT_RETRY_ATTEMPTS,
    ):
        try:
            resolved_inputs = self._resolve_generation_inputs(
                prompt=prompt,
                system_instructions=system_instructions,
                reference_images=reference_images,
                character_pipe=character_pipe,
            )
            request_timeout_seconds = self._normalize_request_timeout_seconds(request_timeout_seconds)
            retry_attempts = self._normalize_retry_attempts(retry_attempts)
        except ValueError as e:
            return self._blank_result(str(e))

        prompt = resolved_inputs["prompt"]
        system_instructions = resolved_inputs["system_instructions"]
        reference_images = resolved_inputs["reference_images"]
        planner_summary = resolved_inputs["planner_summary"]
        character_pipe_connected = resolved_inputs["character_pipe_connected"]
        ignored_direct_inputs = resolved_inputs["ignored_direct_inputs"]

        try:
            prepare_generate_content_request(
                provider_id=self.PROVIDER_ID,
                prompt=prompt,
                model=model,
                seed=seed,
                system_instructions=system_instructions,
                reference_images=reference_images,
                aspect_ratio_override=aspect_ratio_override,
            )
        except GeminiAspectRatioCompatibilityError:
            raise
        except ValueError as e:
            return self._blank_result(str(e))

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
            model_id_for_error = self._resolve_model_id_for_error(model)
            client = self._build_client()
            request = prepare_generate_content_request(
                provider_id=self.PROVIDER_ID,
                prompt=prompt,
                model=model,
                seed=seed,
                system_instructions=system_instructions,
                reference_images=reference_images,
                aspect_ratio_override=aspect_ratio_override,
            )
            model_id_for_error = request.model_id
            response = self._request_generate_content(
                client=client,
                request=request,
                request_timeout_seconds=request_timeout_seconds,
                retry_attempts=retry_attempts,
            )
            parsed_response = parse_generate_content_response(
                response,
                preflight_messages=request.preflight_messages,
            )

            system_messages = self._append_character_pipe_override_note(
                parsed_response.system_messages,
                ignored_direct_inputs,
            )
            system_messages = self._append_planner_summary(
                system_messages,
                planner_summary,
                character_pipe_connected,
            )
            system_messages = ensure_default_text(system_messages, SYSTEM_MESSAGES_DEFAULT)
            generated_image_pipe = self._build_generated_image_pipe(
                parsed_response,
                provider_id=self.PROVIDER_ID,
                model_id=model_id_for_error,
            )

            return (
                parsed_response.image_batch,
                parsed_response.thinking_image_batch,
                parsed_response.thinking_process,
                system_messages,
                generated_image_pipe,
            )

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
            manual_thinking_config = None

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
                config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=image_cfg,
                    tools=tools if tools else None,
                    system_instruction=system_instructions or None,
                    seed=seed % 2147483647 if seed is not None else None,
                )
                manual_thinking_config = {
                    "includeThoughts": flash31_include_thoughts,
                    "thinkingLevel": flash31_thinking_level,
                }
            elif model == "gemini-3-pro-image-preview":
                # Full config: aspect ratio, resolution, text+image
                image_cfg = types.ImageConfig(
                    aspect_ratio=selected_aspect_ratio,
                    image_size=selected_resolution,
                )
                config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=image_cfg,
                    tools=tools if tools else None,
                    system_instruction=system_instructions or None,
                    seed=seed % 2147483647 if seed is not None else None,
                )
            else:
                raise ValueError(f"Unsupported model: {model}")

            # Call API
            response = self._request_generate_content(
                client=client,
                model=model,
                contents=contents,
                config=config,
                manual_thinking_config=manual_thinking_config,
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

            # Thought summaries are only requested for 3.1 Flash Image in this node.
            expects_thought_summary = (
                model == "gemini-3.1-flash-image-preview" and flash31_include_thoughts
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

        except GeminiAspectRatioCompatibilityError:
            raise
        except Exception as e:
            return self._blank_result(
                self._append_planner_summary(
                    self._append_character_pipe_override_note(
                        self._format_generation_error(
                            e,
                            model_id_for_error,
                            request_timeout_seconds,
                        ),
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


class BurveCropMaskLoad:
    DEFAULT_BRUSH_RADIUS_PX = 48.0
    DEFAULT_BRUSH_SOFTNESS = 0.35
    DEFAULT_BRUSH_OPACITY = 1.0
    STATE_VERSION = 1
    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    cls._input_image_options(),
                    {
                        "image_upload": True,
                        "image_folder": "input",
                        "remote": {
                            "route": "/internal/files/input",
                            "refresh_button": True,
                            "control_after_refresh": "first",
                        },
                    },
                ),
                "aspect_ratio": (list(BURVE_CROP_MASK_ASPECT_RATIOS), {"default": "1:1"}),
            },
            "optional": {
                "editor_state_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "advanced": True,
                        "tooltip": "Internal editor state used by the custom crop/mask UI.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "CROP_REGION_PIPE")
    RETURN_NAMES = ("image", "cut_image", "image_mask", "aspect_ratio", "crop_region_pipe")
    FUNCTION = "load"
    CATEGORY = "BurveTools/Image"

    @classmethod
    def _get_input_directory(cls):
        if folder_paths is not None and hasattr(folder_paths, "get_input_directory"):
            return folder_paths.get_input_directory()
        return os.path.join(os.path.dirname(__file__), "input")

    @classmethod
    def _input_image_options(cls):
        images = cls._list_input_images()
        return images or [""]

    @classmethod
    def _list_input_images(cls):
        if folder_paths is not None and hasattr(folder_paths, "get_filename_list"):
            try:
                filenames = folder_paths.get_filename_list("input")
                if filenames:
                    return sorted(filenames)
            except Exception:
                pass

        input_dir = cls._get_input_directory()
        if not os.path.isdir(input_dir):
            return []

        discovered = []
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if os.path.splitext(filename)[1].lower() not in cls.SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, input_dir)
                discovered.append(rel_path.replace("\\", "/"))
        return sorted(discovered)

    @classmethod
    def _sanitize_relative_input_path(cls, image):
        if not isinstance(image, str) or not image.strip():
            raise ValueError("image must be a non-empty string.")
        normalized = image.replace("\\", "/").strip().lstrip("/")
        candidate = os.path.normpath(os.path.join(cls._get_input_directory(), normalized))
        input_dir = os.path.abspath(cls._get_input_directory())
        abs_candidate = os.path.abspath(candidate)
        if os.path.commonpath([input_dir, abs_candidate]) != input_dir:
            raise ValueError("image path escapes the input directory.")
        return abs_candidate

    @classmethod
    def _get_annotated_filepath(cls, image):
        if folder_paths is not None and hasattr(folder_paths, "get_annotated_filepath"):
            return folder_paths.get_annotated_filepath(image)
        return cls._sanitize_relative_input_path(image)

    @classmethod
    def _exists_annotated_filepath(cls, image):
        if folder_paths is not None and hasattr(folder_paths, "exists_annotated_filepath"):
            return folder_paths.exists_annotated_filepath(image)
        try:
            return os.path.isfile(cls._get_annotated_filepath(image))
        except ValueError:
            return False

    @staticmethod
    def _parse_aspect_ratio(aspect_ratio):
        if not isinstance(aspect_ratio, str) or ":" not in aspect_ratio:
            raise ValueError("aspect_ratio must be in W:H format.")
        left, right = aspect_ratio.split(":", 1)
        width = int(left)
        height = int(right)
        if width < 1 or height < 1:
            raise ValueError("aspect_ratio must use positive integers.")
        return width, height

    @classmethod
    def _default_crop_rect(cls, image_width, image_height, aspect_ratio):
        ratio_w, ratio_h = cls._parse_aspect_ratio(aspect_ratio)
        if image_width * ratio_h >= image_height * ratio_w:
            crop_height = image_height
            crop_width = max(1, int(round(crop_height * ratio_w / ratio_h)))
        else:
            crop_width = image_width
            crop_height = max(1, int(round(crop_width * ratio_h / ratio_w)))

        crop_width = min(crop_width, image_width)
        crop_height = min(crop_height, image_height)
        x = max((image_width - crop_width) // 2, 0)
        y = max((image_height - crop_height) // 2, 0)
        return {"x": x, "y": y, "width": crop_width, "height": crop_height}

    @classmethod
    def _ratio_matches(cls, crop, aspect_ratio, tolerance_px=1.0):
        ratio_w, ratio_h = cls._parse_aspect_ratio(aspect_ratio)
        width = float(crop["width"])
        height = float(crop["height"])
        return abs((width * ratio_h) - (height * ratio_w)) <= max(float(ratio_w), float(ratio_h), tolerance_px)

    @classmethod
    def _fit_crop_rect(cls, crop, image_width, image_height, aspect_ratio):
        ratio_w, ratio_h = cls._parse_aspect_ratio(aspect_ratio)
        center_x = float(crop.get("x", 0)) + (float(crop.get("width", image_width)) / 2.0)
        center_y = float(crop.get("y", 0)) + (float(crop.get("height", image_height)) / 2.0)

        if image_width * ratio_h >= image_height * ratio_w:
            max_width = image_height * ratio_w / ratio_h
            max_height = image_height
        else:
            max_width = image_width
            max_height = image_width * ratio_h / ratio_w

        target_width = max(1.0, min(float(crop.get("width", max_width)), float(max_width)))
        target_height = target_width * ratio_h / ratio_w
        if target_height > max_height:
            target_height = float(max_height)
            target_width = target_height * ratio_w / ratio_h

        x = center_x - (target_width / 2.0)
        y = center_y - (target_height / 2.0)
        x = min(max(x, 0.0), max(image_width - target_width, 0.0))
        y = min(max(y, 0.0), max(image_height - target_height, 0.0))

        result = {
            "x": int(round(x)),
            "y": int(round(y)),
            "width": max(1, int(round(target_width))),
            "height": max(1, int(round(target_height))),
        }

        if result["x"] + result["width"] > image_width:
            result["width"] = image_width - result["x"]
        if result["y"] + result["height"] > image_height:
            result["height"] = image_height - result["y"]

        if not cls._ratio_matches(result, aspect_ratio):
            return cls._default_crop_rect(image_width, image_height, aspect_ratio)
        return result

    @staticmethod
    def _default_viewport():
        return {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0}

    @classmethod
    def _default_brush(cls):
        return {
            "radius_px": cls.DEFAULT_BRUSH_RADIUS_PX,
            "softness": cls.DEFAULT_BRUSH_SOFTNESS,
            "opacity": cls.DEFAULT_BRUSH_OPACITY,
            "mode": "paint",
        }

    @classmethod
    def _default_editor_state(cls, image, image_width, image_height, aspect_ratio):
        return {
            "version": cls.STATE_VERSION,
            "image_name": image,
            "source_size": {"width": image_width, "height": image_height},
            "aspect_ratio": aspect_ratio,
            "crop": cls._default_crop_rect(image_width, image_height, aspect_ratio),
            "viewport": cls._default_viewport(),
            "brush": cls._default_brush(),
            "strokes": [],
        }

    @classmethod
    def _parse_editor_state_json(cls, editor_state_json):
        if editor_state_json is None or editor_state_json == "":
            return None
        if not isinstance(editor_state_json, str):
            raise ValueError("editor_state_json must be a string.")
        try:
            payload = json.loads(editor_state_json)
        except json.JSONDecodeError as exc:
            raise ValueError("editor_state_json must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("editor_state_json must decode to an object.")
        version = int(payload.get("version", cls.STATE_VERSION))
        if version != cls.STATE_VERSION:
            raise ValueError("editor_state_json version is unsupported.")
        return payload

    @staticmethod
    def _coerce_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _normalize_brush(cls, brush):
        brush = brush or {}
        mode = str(brush.get("mode", "paint")).lower()
        if mode not in {"paint", "erase"}:
            mode = "paint"
        return {
            "radius_px": max(1.0, cls._coerce_float(brush.get("radius_px"), cls.DEFAULT_BRUSH_RADIUS_PX)),
            "softness": min(max(cls._coerce_float(brush.get("softness"), cls.DEFAULT_BRUSH_SOFTNESS), 0.0), 1.0),
            "opacity": min(max(cls._coerce_float(brush.get("opacity"), cls.DEFAULT_BRUSH_OPACITY), 0.0), 1.0),
            "mode": mode,
        }

    @classmethod
    def _normalize_points(cls, points):
        if not isinstance(points, list):
            raise ValueError("stroke points must be a list.")
        normalized = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("each stroke point must be a [x, y] pair.")
            normalized.append([cls._coerce_float(point[0], 0.0), cls._coerce_float(point[1], 0.0)])
        if not normalized:
            raise ValueError("stroke must contain at least one point.")
        return normalized

    @classmethod
    def _normalize_strokes(cls, strokes):
        if strokes is None:
            return []
        if not isinstance(strokes, list):
            raise ValueError("strokes must be a list.")
        normalized = []
        for stroke in strokes:
            if not isinstance(stroke, dict):
                raise ValueError("each stroke must be an object.")
            brush = cls._normalize_brush(stroke)
            normalized.append(
                {
                    "mode": brush["mode"],
                    "radius_px": brush["radius_px"],
                    "softness": brush["softness"],
                    "opacity": brush["opacity"],
                    "points": cls._normalize_points(stroke.get("points", [])),
                }
            )
        return normalized

    @classmethod
    def _reconcile_editor_state(cls, image, image_width, image_height, aspect_ratio, editor_state_json=""):
        parsed = cls._parse_editor_state_json(editor_state_json)
        if parsed is None:
            return cls._default_editor_state(image, image_width, image_height, aspect_ratio)

        source_size = parsed.get("source_size", {})
        source_width = int(source_size.get("width", 0) or 0)
        source_height = int(source_size.get("height", 0) or 0)
        state_image = parsed.get("image_name")
        if (
            state_image != image
            or source_width != image_width
            or source_height != image_height
        ):
            return cls._default_editor_state(image, image_width, image_height, aspect_ratio)

        crop = parsed.get("crop")
        if not isinstance(crop, dict):
            raise ValueError("editor_state_json crop must be an object.")
        normalized_crop = {
            "x": int(round(cls._coerce_float(crop.get("x"), 0))),
            "y": int(round(cls._coerce_float(crop.get("y"), 0))),
            "width": int(round(cls._coerce_float(crop.get("width"), image_width))),
            "height": int(round(cls._coerce_float(crop.get("height"), image_height))),
        }
        normalized_crop = cls._fit_crop_rect(normalized_crop, image_width, image_height, aspect_ratio)

        viewport = parsed.get("viewport") or {}
        brush = cls._normalize_brush(parsed.get("brush"))
        strokes = cls._normalize_strokes(parsed.get("strokes"))

        return {
            "version": cls.STATE_VERSION,
            "image_name": image,
            "source_size": {"width": image_width, "height": image_height},
            "aspect_ratio": aspect_ratio,
            "crop": normalized_crop,
            "viewport": {
                "zoom": max(cls._coerce_float(viewport.get("zoom"), 1.0), 0.1),
                "pan_x": cls._coerce_float(viewport.get("pan_x"), 0.0),
                "pan_y": cls._coerce_float(viewport.get("pan_y"), 0.0),
            },
            "brush": brush,
            "strokes": strokes,
        }

    @classmethod
    def _validate_crop_rect(cls, crop, image_width, image_height, aspect_ratio):
        if crop["width"] < 1 or crop["height"] < 1:
            raise ValueError("crop dimensions must be positive.")
        if crop["x"] < 0 or crop["y"] < 0:
            raise ValueError("crop origin must be inside the image.")
        if crop["x"] + crop["width"] > image_width or crop["y"] + crop["height"] > image_height:
            raise ValueError("crop must stay inside the source image.")
        if not cls._ratio_matches(crop, aspect_ratio):
            raise ValueError("crop does not match the selected aspect ratio.")

    @classmethod
    def _load_source_image(cls, image):
        image_path = cls._get_annotated_filepath(image)
        with Image.open(image_path) as pil_image:
            rgb_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        return rgb_image

    @staticmethod
    def _pil_to_image_tensor(pil_image):
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        return torch.from_numpy(image_np).unsqueeze(0)

    @staticmethod
    def _mask_to_tensor(mask_array):
        return torch.from_numpy(mask_array.astype(np.float32)).unsqueeze(0)

    @classmethod
    def _render_mask_strokes(cls, image_width, image_height, crop, strokes):
        mask = np.zeros((image_height, image_width), dtype=np.float32)
        if not strokes:
            return mask

        crop_left = int(crop["x"])
        crop_top = int(crop["y"])
        crop_right = crop_left + int(crop["width"])
        crop_bottom = crop_top + int(crop["height"])

        for stroke in strokes:
            points = stroke["points"]
            radius = max(float(stroke["radius_px"]), 1.0)
            softness = min(max(float(stroke["softness"]), 0.0), 1.0)
            opacity = min(max(float(stroke["opacity"]), 0.0), 1.0)
            mode = stroke["mode"]

            if len(points) == 1:
                cls._apply_brush_dab(
                    mask,
                    points[0][0],
                    points[0][1],
                    radius,
                    softness,
                    opacity,
                    mode,
                    crop_left,
                    crop_top,
                    crop_right,
                    crop_bottom,
                )
                continue

            step = max(radius * 0.25, 1.0)
            for start, end in zip(points, points[1:]):
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                distance = math.hypot(dx, dy)
                if distance <= 0.0:
                    cls._apply_brush_dab(
                        mask,
                        start[0],
                        start[1],
                        radius,
                        softness,
                        opacity,
                        mode,
                        crop_left,
                        crop_top,
                        crop_right,
                        crop_bottom,
                    )
                    continue
                steps = max(int(math.ceil(distance / step)), 1)
                for index in range(steps + 1):
                    t = index / steps
                    x = start[0] + (dx * t)
                    y = start[1] + (dy * t)
                    cls._apply_brush_dab(
                        mask,
                        x,
                        y,
                        radius,
                        softness,
                        opacity,
                        mode,
                        crop_left,
                        crop_top,
                        crop_right,
                        crop_bottom,
                    )

        return mask

    @staticmethod
    def _apply_brush_dab(
        mask,
        center_x,
        center_y,
        radius,
        softness,
        opacity,
        mode,
        crop_left,
        crop_top,
        crop_right,
        crop_bottom,
    ):
        left = max(int(math.floor(center_x - radius)), crop_left)
        top = max(int(math.floor(center_y - radius)), crop_top)
        right = min(int(math.ceil(center_x + radius)) + 1, crop_right)
        bottom = min(int(math.ceil(center_y + radius)) + 1, crop_bottom)
        if left >= right or top >= bottom:
            return

        ys, xs = np.ogrid[top:bottom, left:right]
        distances = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
        alpha = np.zeros((bottom - top, right - left), dtype=np.float32)
        if softness <= 0.0:
            alpha[distances <= radius] = opacity
        else:
            core_radius = radius * max(0.0, 1.0 - softness)
            if core_radius > 0.0:
                alpha[distances <= core_radius] = opacity
            feather = max(radius - core_radius, 1e-6)
            falloff_mask = (distances > core_radius) & (distances <= radius)
            if np.any(falloff_mask):
                normalized = 1.0 - ((distances[falloff_mask] - core_radius) / feather)
                alpha[falloff_mask] = opacity * np.clip(normalized ** 2, 0.0, 1.0)

        if mode == "erase":
            mask[top:bottom, left:right] *= (1.0 - alpha)
        else:
            mask[top:bottom, left:right] = np.maximum(mask[top:bottom, left:right], alpha)

    @classmethod
    def _build_crop_region_pipe(cls, image_width, image_height, aspect_ratio, crop):
        return {
            "kind": CROP_REGION_PIPE_KIND,
            "version": CROP_REGION_PIPE_VERSION,
            "source_width": int(image_width),
            "source_height": int(image_height),
            "aspect_ratio": aspect_ratio,
            "crop": {
                "x": int(crop["x"]),
                "y": int(crop["y"]),
                "width": int(crop["width"]),
                "height": int(crop["height"]),
            },
        }

    @classmethod
    def _normalize_crop_region_pipe(cls, crop_region_pipe):
        if crop_region_pipe is None:
            return None
        if not isinstance(crop_region_pipe, dict):
            raise ValueError("Invalid crop_region_pipe: expected a dict payload.")
        if crop_region_pipe.get("kind") != CROP_REGION_PIPE_KIND:
            raise ValueError("Invalid crop_region_pipe: unexpected kind.")
        if crop_region_pipe.get("version") != CROP_REGION_PIPE_VERSION:
            raise ValueError("Invalid crop_region_pipe: unsupported version.")
        crop = crop_region_pipe.get("crop")
        if not isinstance(crop, dict):
            raise ValueError("Invalid crop_region_pipe: crop must be a dict.")
        return {
            "kind": CROP_REGION_PIPE_KIND,
            "version": CROP_REGION_PIPE_VERSION,
            "source_width": int(crop_region_pipe.get("source_width", 0)),
            "source_height": int(crop_region_pipe.get("source_height", 0)),
            "aspect_ratio": str(crop_region_pipe.get("aspect_ratio", "")),
            "crop": {
                "x": int(crop.get("x", 0)),
                "y": int(crop.get("y", 0)),
                "width": int(crop.get("width", 0)),
                "height": int(crop.get("height", 0)),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image, aspect_ratio, editor_state_json="", **kwargs):
        if aspect_ratio not in BURVE_CROP_MASK_ASPECT_RATIOS:
            return "aspect_ratio must be one of the supported Burve crop ratios."
        if not cls._exists_annotated_filepath(image):
            return f"Input image not found: {image}"
        try:
            pil_image = cls._load_source_image(image)
            image_width, image_height = pil_image.size
            parsed = cls._parse_editor_state_json(editor_state_json)
            if parsed is not None:
                source_size = parsed.get("source_size", {})
                if (
                    parsed.get("image_name") == image
                    and int(source_size.get("width", 0) or 0) == image_width
                    and int(source_size.get("height", 0) or 0) == image_height
                ):
                    crop = parsed.get("crop")
                    if not isinstance(crop, dict):
                        raise ValueError("editor_state_json crop must be an object.")
                    raw_crop = {
                        "x": int(round(cls._coerce_float(crop.get("x"), 0))),
                        "y": int(round(cls._coerce_float(crop.get("y"), 0))),
                        "width": int(round(cls._coerce_float(crop.get("width"), image_width))),
                        "height": int(round(cls._coerce_float(crop.get("height"), image_height))),
                    }
                    cls._validate_crop_rect(raw_crop, image_width, image_height, aspect_ratio)
                    cls._normalize_strokes(parsed.get("strokes"))
            state = cls._reconcile_editor_state(image, image_width, image_height, aspect_ratio, editor_state_json)
            cls._validate_crop_rect(state["crop"], image_width, image_height, aspect_ratio)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, image, aspect_ratio, editor_state_json="", **kwargs):
        image_path = cls._get_annotated_filepath(image)
        hasher = hashlib.sha256()
        with open(image_path, "rb") as handle:
            hasher.update(handle.read())
        hasher.update(aspect_ratio.encode("utf-8"))
        hasher.update(str(editor_state_json or "").encode("utf-8"))
        return hasher.hexdigest()

    def load(self, image, aspect_ratio, editor_state_json=""):
        pil_image = self._load_source_image(image)
        image_width, image_height = pil_image.size
        state = self._reconcile_editor_state(image, image_width, image_height, aspect_ratio, editor_state_json)
        self._validate_crop_rect(state["crop"], image_width, image_height, aspect_ratio)

        crop = state["crop"]
        crop_box = (
            int(crop["x"]),
            int(crop["y"]),
            int(crop["x"] + crop["width"]),
            int(crop["y"] + crop["height"]),
        )
        cropped_image = pil_image.crop(crop_box)
        image_mask = self._render_mask_strokes(image_width, image_height, crop, state["strokes"])
        crop_region_pipe = self._build_crop_region_pipe(image_width, image_height, aspect_ratio, crop)

        return (
            self._pil_to_image_tensor(pil_image),
            self._pil_to_image_tensor(cropped_image),
            self._mask_to_tensor(image_mask),
            aspect_ratio,
            crop_region_pipe,
        )


class BurveCropMaskApply:
    MAX_AUTO_FIT_TRIM_FRACTION = 0.01

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "new_image": ("IMAGE",),
                "crop_region_pipe": ("CROP_REGION_PIPE",),
                "mask": ("MASK",),
                "strict_aspect_ratio": ("BOOLEAN", {"default": False, "advanced": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "placed_image_white_bg", "masked_placed_image_white_bg", "status")
    FUNCTION = "apply"
    CATEGORY = "BurveTools/Image"

    @staticmethod
    def _require_single_image_tensor(name, image):
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 4:
            raise ValueError(f"[BurveCropMaskApply] {name} must be an IMAGE tensor with shape (B, H, W, C).")
        if int(shape[0]) != 1:
            raise ValueError(f"[BurveCropMaskApply] {name} must have batch size 1.")
        if int(shape[1]) < 1 or int(shape[2]) < 1:
            raise ValueError(f"[BurveCropMaskApply] {name} must have positive spatial dimensions.")
        return int(shape[1]), int(shape[2])

    @staticmethod
    def _require_single_mask_tensor(mask):
        shape = getattr(mask, "shape", None)
        if shape is None or len(shape) != 3:
            raise ValueError("[BurveCropMaskApply] mask must be a MASK tensor with shape (B, H, W).")
        if int(shape[0]) != 1:
            raise ValueError("[BurveCropMaskApply] mask must have batch size 1.")
        if int(shape[1]) < 1 or int(shape[2]) < 1:
            raise ValueError("[BurveCropMaskApply] mask must have positive spatial dimensions.")
        return int(shape[1]), int(shape[2])

    @staticmethod
    def _format_dimension_aspect_ratio(width, height):
        divisor = math.gcd(int(width), int(height)) or 1
        return f"{int(width) // divisor}:{int(height) // divisor}"

    @classmethod
    def _aspect_crop_loss_fraction(cls, width, height, aspect_ratio):
        ratio_w, ratio_h = BurveCropMaskLoad._parse_aspect_ratio(aspect_ratio)
        source_ratio = float(width) / float(height)
        target_ratio = float(ratio_w) / float(ratio_h)
        if source_ratio >= target_ratio:
            return max(0.0, 1.0 - (target_ratio / source_ratio))
        return max(0.0, 1.0 - (source_ratio / target_ratio))

    @staticmethod
    def _resize_cover_and_center_crop(image, target_height, target_width):
        image_bchw = image.permute(0, 3, 1, 2)
        source_height = int(image_bchw.shape[2])
        source_width = int(image_bchw.shape[3])

        if source_width * target_height == source_height * target_width:
            resized = F.interpolate(
                image_bchw,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            return resized.permute(0, 2, 3, 1)

        if source_width * target_height > source_height * target_width:
            resize_height = target_height
            resize_width = max(1, int(math.ceil((target_height * source_width) / float(source_height))))
        else:
            resize_width = target_width
            resize_height = max(1, int(math.ceil((target_width * source_height) / float(source_width))))

        resized = F.interpolate(
            image_bchw,
            size=(resize_height, resize_width),
            mode="bilinear",
            align_corners=False,
        )

        crop_y = max((resize_height - target_height) // 2, 0)
        crop_x = max((resize_width - target_width) // 2, 0)
        cropped = resized[:, :, crop_y:crop_y + target_height, crop_x:crop_x + target_width]
        return cropped.permute(0, 2, 3, 1)

    @classmethod
    def _format_aspect_ratio_message(cls, width, height, aspect_ratio, trim_fraction, kind):
        actual_ratio = cls._format_dimension_aspect_ratio(width, height)
        prefix = f"[BurveCropMaskApply] new_image {int(width)}x{int(height)} ({actual_ratio})"
        if kind == "warning":
            return (
                f"{prefix} was normalized to {aspect_ratio} by centered crop; "
                f"estimated trim {trim_fraction * 100.0:.2f}%."
            )
        if kind == "strict_error":
            return f"{prefix} does not match required {aspect_ratio} and strict_aspect_ratio is enabled."
        return (
            f"{prefix} differs too much from required {aspect_ratio} for safe auto-fit; "
            f"estimated trim {trim_fraction * 100.0:.2f}% exceeds "
            f"{cls.MAX_AUTO_FIT_TRIM_FRACTION * 100.0:.2f}% limit."
        )

    def apply(self, base_image, new_image, crop_region_pipe, mask, strict_aspect_ratio=False):
        base_height, base_width = self._require_single_image_tensor("base_image", base_image)
        new_height, new_width = self._require_single_image_tensor("new_image", new_image)
        mask_height, mask_width = self._require_single_mask_tensor(mask)

        normalized_pipe = BurveCropMaskLoad._normalize_crop_region_pipe(crop_region_pipe)
        BurveCropMaskLoad._validate_crop_rect(
            normalized_pipe["crop"],
            normalized_pipe["source_width"],
            normalized_pipe["source_height"],
            normalized_pipe["aspect_ratio"],
        )

        if normalized_pipe["source_width"] != base_width or normalized_pipe["source_height"] != base_height:
            raise ValueError("[BurveCropMaskApply] crop_region_pipe source size must match base_image dimensions.")
        if mask_height != base_height or mask_width != base_width:
            raise ValueError("[BurveCropMaskApply] mask dimensions must match base_image dimensions.")

        base_image = base_image
        new_image = new_image.to(device=base_image.device, dtype=base_image.dtype)
        mask = mask.to(device=base_image.device, dtype=base_image.dtype)
        status = ""

        crop = normalized_pipe["crop"]
        crop_x = int(crop["x"])
        crop_y = int(crop["y"])
        crop_width = int(crop["width"])
        crop_height = int(crop["height"])

        if BurveCropMaskLoad._ratio_matches(
            {"width": new_width, "height": new_height},
            normalized_pipe["aspect_ratio"],
        ):
            resized_new_image = F.interpolate(
                new_image.permute(0, 3, 1, 2),
                size=(crop_height, crop_width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            trim_fraction = self._aspect_crop_loss_fraction(new_width, new_height, normalized_pipe["aspect_ratio"])
            if strict_aspect_ratio:
                raise ValueError(
                    self._format_aspect_ratio_message(
                        new_width,
                        new_height,
                        normalized_pipe["aspect_ratio"],
                        trim_fraction,
                        "strict_error",
                    )
                )
            if trim_fraction > self.MAX_AUTO_FIT_TRIM_FRACTION:
                raise ValueError(
                    self._format_aspect_ratio_message(
                        new_width,
                        new_height,
                        normalized_pipe["aspect_ratio"],
                        trim_fraction,
                        "limit_error",
                    )
                )
            resized_new_image = self._resize_cover_and_center_crop(new_image, crop_height, crop_width)
            status = self._format_aspect_ratio_message(
                new_width,
                new_height,
                normalized_pipe["aspect_ratio"],
                trim_fraction,
                "warning",
            )

        white_canvas = torch.ones_like(base_image)
        placed_image_white_bg = white_canvas.clone()
        placed_image_white_bg[:, crop_y:crop_y + crop_height, crop_x:crop_x + crop_width, :] = resized_new_image

        crop_gate = torch.zeros_like(mask)
        crop_gate[:, crop_y:crop_y + crop_height, crop_x:crop_x + crop_width] = 1.0
        effective_mask = torch.clamp(mask, 0.0, 1.0) * crop_gate
        mask4d = effective_mask.unsqueeze(-1)

        edited_image = base_image * (1.0 - mask4d) + placed_image_white_bg * mask4d
        masked_placed_image_white_bg = white_canvas * (1.0 - mask4d) + placed_image_white_bg * mask4d
        return (edited_image, placed_image_white_bg, masked_placed_image_white_bg, status)


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


class BurveImageInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("info", "width", "height", "aspect_ratio")
    FUNCTION = "inspect"
    CATEGORY = "BurveTools"
    OUTPUT_NODE = True

    def inspect(self, image):
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 4:
            raise ValueError("[BurveImageInfo] Expected IMAGE tensor with shape (B, H, W, C).")

        height = int(shape[1])
        width = int(shape[2])

        if height < 1 or width < 1:
            raise ValueError("[BurveImageInfo] Received empty image tensor.")

        divisor = math.gcd(width, height) or 1
        aspect_ratio = f"{width // divisor}:{height // divisor}"
        info = f"Size: {width}x{height} px\nAspect ratio: {aspect_ratio}"
        return (info, width, height, aspect_ratio)


class BurveSaveGeneratedImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Burve"}),
            },
            "optional": {
                "generated_image_pipe": ("GENERATED_IMAGE_PIPE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)
    FUNCTION = "save"
    CATEGORY = "BurveTools"
    OUTPUT_NODE = True

    def _mime_type_to_extension(self, mime_type):
        return {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
        }.get(mime_type)

    def _normalize_generated_image_pipe(self, generated_image_pipe):
        if generated_image_pipe is None:
            return None
        if not isinstance(generated_image_pipe, dict):
            raise ValueError("Invalid generated_image_pipe: expected a dict payload.")
        if generated_image_pipe.get("kind") != GENERATED_IMAGE_PIPE_KIND:
            raise ValueError("Invalid generated_image_pipe: unexpected kind.")
        if generated_image_pipe.get("version") != GENERATED_IMAGE_PIPE_VERSION:
            raise ValueError("Invalid generated_image_pipe: unsupported version.")
        items = generated_image_pipe.get("items", [])
        if not isinstance(items, list):
            raise ValueError("Invalid generated_image_pipe: items must be a list.")
        notes = generated_image_pipe.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("Invalid generated_image_pipe: notes must be a string.")
        return generated_image_pipe

    def _get_output_directory(self):
        if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _sanitize_prefix(self, filename_prefix):
        safe = str(filename_prefix or "Burve").strip().replace("\\", "/")
        if not safe:
            return "Burve"
        parts = [part for part in safe.split("/") if part not in {"", "."}]
        sanitized_parts = []
        for part in parts:
            sanitized_part = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in part)
            if sanitized_part:
                sanitized_parts.append(sanitized_part)
        if not sanitized_parts:
            return "Burve"
        return "/".join(sanitized_parts)

    def _fallback_get_save_image_path(self, filename_prefix, image_width=0, image_height=0):
        output_dir = self._get_output_directory()
        normalized_prefix = self._sanitize_prefix(filename_prefix)
        subfolder = os.path.dirname(os.path.normpath(normalized_prefix))
        if subfolder in {"", "."}:
            subfolder = ""
        filename = os.path.basename(os.path.normpath(normalized_prefix)) or "Burve"
        full_output_folder = os.path.join(output_dir, subfolder) if subfolder else output_dir
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
        while True:
            prefix = f"{filename}_{counter:05d}_"
            if not any(name.startswith(prefix) for name in os.listdir(full_output_folder)):
                break
            counter += 1
        return full_output_folder, filename, counter, subfolder, normalized_prefix

    def _get_save_image_path(self, filename_prefix, image_width=0, image_height=0):
        output_dir = self._get_output_directory()
        if folder_paths is not None and hasattr(folder_paths, "get_save_image_path"):
            return folder_paths.get_save_image_path(filename_prefix, output_dir, image_width, image_height)
        return self._fallback_get_save_image_path(filename_prefix, image_width, image_height)

    def _build_png_metadata(self, prompt=None, extra_pnginfo=None):
        if PngInfo is None:
            return None
        if prompt is None and not isinstance(extra_pnginfo, dict):
            return None
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if isinstance(extra_pnginfo, dict):
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value))
        return metadata

    def _save_png_fallback(self, frame, output_path, prompt=None, extra_pnginfo=None):
        image_np = (frame.cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)
        metadata = self._build_png_metadata(prompt=prompt, extra_pnginfo=extra_pnginfo)
        save_kwargs = {"format": "PNG"}
        if metadata is not None:
            save_kwargs["pnginfo"] = metadata
        Image.fromarray(image_np).save(output_path, **save_kwargs)
        return output_path

    def save(self, image, filename_prefix="Burve", generated_image_pipe=None, prompt=None, extra_pnginfo=None):
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 4:
            raise ValueError("[BurveSaveGeneratedImage] Expected IMAGE tensor with shape (B, H, W, C).")

        normalized_pipe = self._normalize_generated_image_pipe(generated_image_pipe)
        batch_size = int(shape[0])
        lines = []
        ui_images = []
        mismatch = False
        if normalized_pipe is not None and len(normalized_pipe.get("items", [])) != batch_size:
            mismatch = True
            lines.append("generated_image_pipe item count did not match the image batch; saved all frames as PNG fallback.")

        full_output_folder, filename, counter, subfolder, _ = self._get_save_image_path(
            filename_prefix,
            int(shape[2]),
            int(shape[1]),
        )

        for index in range(batch_size):
            frame = image[index]
            item = None
            if normalized_pipe is not None and not mismatch:
                item = normalized_pipe["items"][index]

            filename_with_batch_num = filename.replace("%batch_num%", str(index))
            mime_type = item.get("mime_type") if item else None
            extension = self._mime_type_to_extension(mime_type) if mime_type else None
            raw_bytes = item.get("raw_bytes") if item else None

            if raw_bytes is not None and extension is not None:
                file = f"{filename_with_batch_num}_{counter:05}_.{extension}"
                output_path = os.path.join(full_output_folder, file)
                with open(output_path, "wb") as handle:
                    handle.write(raw_bytes)
                lines.append(output_path)
            else:
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                output_path = os.path.join(full_output_folder, file)
                self._save_png_fallback(
                    frame,
                    output_path,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
                if item is not None and item.get("raw_bytes") is None:
                    lines.append(f"{output_path}\nframe {index + 1}: raw bytes unavailable, saved PNG fallback")
                elif item is not None and extension is None and mime_type:
                    lines.append(f"{output_path}\nframe {index + 1}: unsupported mime_type {mime_type}, saved PNG fallback")
                else:
                    lines.append(output_path)
            ui_images.append(
                {
                    "filename": file,
                    "subfolder": subfolder,
                    "type": "output",
                }
            )
            counter += 1

        if normalized_pipe is not None and normalized_pipe.get("notes"):
            lines.append(normalized_pipe["notes"])

        return {
            "ui": {"images": ui_images},
            "result": ("\n".join(lines),),
        }


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
