from __future__ import annotations

import hashlib
import importlib.metadata
import io
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types

MIN_GOOGLE_GENAI_VERSION = "1.68.0"
THINKING_PROCESS_DEFAULT = "No thinking output returned by the model."
SYSTEM_MESSAGES_DEFAULT = "No system messages."
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_RETRY_ATTEMPTS = 5
RETRYABLE_STATUS_CODES = (408, 429, 500, 502, 503, 504)
BACKOFF_INITIAL_DELAY_SECONDS = 1.0
BACKOFF_EXP_BASE = 2.0
BACKOFF_MAX_DELAY_SECONDS = 16.0
BACKOFF_JITTER_MIN = 0.8
BACKOFF_JITTER_MAX = 1.2
NO_IMAGE_MESSAGES = (
    "No non-thinking image generated.",
    "The request completed, but no usable image part was found in the response.",
)


@dataclass(frozen=True)
class GeminiImageFieldSpec:
    key: str
    field_type: str
    default: Any
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeminiImageModelSpec:
    model_id: str
    fields: tuple[GeminiImageFieldSpec, ...]


@dataclass(frozen=True)
class NormalizedModelSelection:
    model_id: str
    values: MappingProxyType


@dataclass(frozen=True)
class PreparedGeminiRequest:
    model_id: str
    contents: list[Any]
    config: Any
    preflight_messages: tuple[str, ...]


class GeminiAspectRatioCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedGeminiResponse:
    image_batch: torch.Tensor
    thinking_image_batch: torch.Tensor
    thinking_process: str
    system_messages: str
    image_artifacts: tuple["ParsedGeminiImageArtifact", ...]


@dataclass(frozen=True)
class ParsedGeminiImageArtifact:
    mime_type: str | None
    raw_bytes: bytes
    sha256: str
    source: str


def _combo_field(key: str, default: str, options: tuple[str, ...]) -> GeminiImageFieldSpec:
    return GeminiImageFieldSpec(key=key, field_type="combo", default=default, options=options)


def _bool_field(key: str, default: bool) -> GeminiImageFieldSpec:
    return GeminiImageFieldSpec(key=key, field_type="boolean", default=default)


GEMINI_IMAGE_MODEL_SPECS = MappingProxyType(
    {
        "gemini-2.5-flash-image": GeminiImageModelSpec(
            model_id="gemini-2.5-flash-image",
            fields=(
                _combo_field(
                    "aspect_ratio",
                    "1:1",
                    ("1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"),
                ),
            ),
        ),
        "gemini-3-pro-image-preview": GeminiImageModelSpec(
            model_id="gemini-3-pro-image-preview",
            fields=(
                _combo_field(
                    "aspect_ratio",
                    "1:1",
                    ("1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"),
                ),
                _combo_field("resolution", "1K", ("1K", "2K", "4K")),
                _combo_field("search_mode", "off", ("off", "web")),
            ),
        ),
        "gemini-3.1-flash-image-preview": GeminiImageModelSpec(
            model_id="gemini-3.1-flash-image-preview",
            fields=(
                _combo_field(
                    "aspect_ratio",
                    "1:1",
                    ("1:1", "3:2", "2:3", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"),
                ),
                _combo_field("resolution", "1K", ("512", "1K", "2K", "4K")),
                _combo_field("search_mode", "off", ("off", "web", "image", "web+image")),
                _combo_field("thinking_level", "HIGH", ("MINIMAL", "HIGH")),
                _bool_field("include_thoughts", False),
                _combo_field("output_mime_type", "image/png", ("image/png", "image/jpeg", "image/webp")),
                _combo_field("prominent_people", "allow", ("allow", "block")),
            ),
        ),
    }
)

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def blank_image_tensor() -> torch.Tensor:
    return torch.zeros((1, 64, 64, 3))


def get_google_genai_version() -> str:
    version = getattr(genai, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version.strip()
    try:
        return importlib.metadata.version("google-genai")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _parse_version(version: str) -> tuple[int, int, int]:
    match = _VERSION_RE.search(str(version))
    if match is None:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def ensure_google_genai_compatibility() -> None:
    version = get_google_genai_version()
    parsed = _parse_version(version)
    missing_symbols = [
        symbol
        for symbol in (
            "ThinkingConfig",
            "ThinkingLevel",
            "SearchTypes",
            "WebSearch",
            "ImageSearch",
            "ProminentPeople",
        )
        if not hasattr(types, symbol)
    ]

    if parsed < (1, 68, 0) or parsed >= (2, 0, 0) or missing_symbols:
        details = [f"detected google-genai version: {version}"]
        if missing_symbols:
            details.append(f"missing symbols: {', '.join(missing_symbols)}")
        raise RuntimeError(
            "ComfyUI_Burve_Tools 2.1.0 requires google-genai>=1.68.0,<2 for the "
            "Gemini DynamicCombo nodes.\n" + "\n".join(details)
        )


def get_model_spec(model_id: str) -> GeminiImageModelSpec:
    try:
        return GEMINI_IMAGE_MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_id}") from exc


def get_field_default(model_id: str, field_key: str) -> Any:
    spec = get_model_spec(model_id)
    for field in spec.fields:
        if field.key == field_key:
            return field.default
    raise KeyError(field_key)


def build_dynamic_combo_options(io_module) -> list[Any]:
    options = []
    for model_id, spec in GEMINI_IMAGE_MODEL_SPECS.items():
        inputs = []
        for field in spec.fields:
            if field.field_type == "combo":
                inputs.append(
                    io_module.Combo.Input(
                        field.key,
                        options=list(field.options),
                        default=field.default,
                    )
                )
            elif field.field_type == "boolean":
                inputs.append(io_module.Boolean.Input(field.key, default=field.default))
            else:
                raise ValueError(f"Unsupported Gemini model field type: {field.field_type}")
        options.append(io_module.DynamicCombo.Option(model_id, inputs))
    return options


def normalize_model_selection(model: Any) -> NormalizedModelSelection:
    if isinstance(model, str):
        model_id = model
        provided_values = {}
    elif isinstance(model, dict):
        model_id = model.get("model")
        provided_values = {key: value for key, value in model.items() if key != "model"}
    else:
        raise ValueError("Model input must be a DynamicCombo payload or a model id string.")

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Model input is missing the selected model id.")

    spec = get_model_spec(model_id)
    allowed_keys = {field.key for field in spec.fields}
    unexpected_keys = sorted(set(provided_values) - allowed_keys)
    if unexpected_keys:
        raise ValueError(
            f"Unsupported parameters for {model_id}: {', '.join(unexpected_keys)}."
        )

    normalized_values: dict[str, Any] = {}
    for field in spec.fields:
        value = provided_values.get(field.key, field.default)
        if field.field_type == "combo":
            if not isinstance(value, str):
                raise ValueError(f"{field.key} must be a string for {model_id}.")
            if value not in field.options:
                raise ValueError(
                    f"Invalid value for {field.key} on {model_id}. "
                    f"Expected one of: {', '.join(field.options)}."
                )
        elif field.field_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{field.key} must be a boolean for {model_id}.")
        normalized_values[field.key] = value

    return NormalizedModelSelection(
        model_id=model_id,
        values=MappingProxyType(normalized_values),
    )


def resolve_model_selection_with_aspect_ratio_override(
    model: Any,
    aspect_ratio_override: Any = "",
) -> NormalizedModelSelection:
    selection = normalize_model_selection(model)

    if aspect_ratio_override is None:
        return selection
    if not isinstance(aspect_ratio_override, str):
        raise GeminiAspectRatioCompatibilityError("aspect_ratio_override must be a string.")

    normalized_override = aspect_ratio_override.strip()
    if not normalized_override:
        return selection

    spec = get_model_spec(selection.model_id)
    aspect_ratio_field = next((field for field in spec.fields if field.key == "aspect_ratio"), None)
    if aspect_ratio_field is None:
        raise GeminiAspectRatioCompatibilityError(
            f"Selected model {selection.model_id} does not support aspect_ratio '{normalized_override}'."
        )

    if normalized_override not in aspect_ratio_field.options:
        supported_values = ", ".join(aspect_ratio_field.options)
        raise GeminiAspectRatioCompatibilityError(
            f"Selected model {selection.model_id} does not support aspect_ratio "
            f"'{normalized_override}'. Supported values: {supported_values}."
        )

    resolved_values = dict(selection.values)
    resolved_values["aspect_ratio"] = normalized_override
    return NormalizedModelSelection(
        model_id=selection.model_id,
        values=MappingProxyType(resolved_values),
    )


def prepare_generate_content_request(
    *,
    provider_id: str,
    prompt: str,
    model: Any,
    seed: int,
    system_instructions: str = "",
    reference_images: Any = None,
    aspect_ratio_override: Any = "",
) -> PreparedGeminiRequest:
    selection = resolve_model_selection_with_aspect_ratio_override(
        model,
        aspect_ratio_override=aspect_ratio_override,
    )
    spec = get_model_spec(selection.model_id)
    preflight_messages: list[str] = []

    image_config_kwargs = {
        "aspect_ratio": selection.values["aspect_ratio"],
    }

    if "resolution" in selection.values:
        image_config_kwargs["image_size"] = selection.values["resolution"]

    if selection.model_id == "gemini-3.1-flash-image-preview":
        if provider_id == "vertex":
            image_config_kwargs["output_mime_type"] = selection.values["output_mime_type"]
            image_config_kwargs["prominent_people"] = _enum_value(
                types.ProminentPeople,
                {
                    "allow": "ALLOW_PROMINENT_PEOPLE",
                    "block": "BLOCK_PROMINENT_PEOPLE",
                }[selection.values["prominent_people"]],
            )
        else:
            if selection.values["output_mime_type"] != get_field_default(spec.model_id, "output_mime_type"):
                preflight_messages.append(
                    "output_mime_type is not supported by the Gemini API SDK path. Ignoring selected value."
                )
            if selection.values["prominent_people"] != get_field_default(spec.model_id, "prominent_people"):
                preflight_messages.append(
                    "prominent_people is not supported by the Gemini API SDK path. Ignoring selected value."
                )

    config_kwargs = {
        "response_modalities": ["TEXT", "IMAGE"],
        "image_config": types.ImageConfig(**image_config_kwargs),
        "system_instruction": system_instructions or None,
        "seed": seed % 2147483647 if seed is not None else None,
    }

    tools = _build_tools(selection)
    if tools:
        config_kwargs["tools"] = tools

    thinking_config = _build_thinking_config(selection)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config

    return PreparedGeminiRequest(
        model_id=selection.model_id,
        contents=build_contents(prompt=prompt, reference_images=reference_images),
        config=types.GenerateContentConfig(**config_kwargs),
        preflight_messages=tuple(preflight_messages),
    )


def clone_config_with_http_timeout(config: Any, timeout_ms: int) -> Any:
    config_kwargs = dict(vars(config))
    config_kwargs["http_options"] = types.HttpOptions(timeout=int(timeout_ms))
    return types.GenerateContentConfig(**config_kwargs)


def build_contents(*, prompt: str, reference_images: Any = None) -> list[Any]:
    contents: list[Any] = [prompt]

    if reference_images is None:
        return contents

    for img_tensor in reference_images:
        if img_tensor is None or not isinstance(img_tensor, torch.Tensor):
            continue

        batch = img_tensor.shape[0]
        for index in range(batch):
            if len(contents) - 1 >= 14:
                return contents
            frame = img_tensor[index]
            image_np = (frame.cpu().numpy() * 255).astype(np.uint8)
            contents.append(Image.fromarray(image_np))

    return contents


def parse_generate_content_response(
    response: Any,
    preflight_messages: tuple[str, ...] = (),
) -> ParsedGeminiResponse:
    normal_images: list[torch.Tensor] = []
    normal_image_artifacts: list[ParsedGeminiImageArtifact] = []
    thinking_images: list[torch.Tensor] = []
    thought_chunks: list[str] = []
    answer_chunks: list[str] = []
    system_messages = "\n".join(message for message in preflight_messages if message).strip()

    candidates = getattr(response, "candidates", None)
    parts = []
    first_candidate = None
    if candidates:
        first_candidate = candidates[0]
        content = getattr(first_candidate, "content", None)
        if content is not None:
            parts = getattr(content, "parts", None) or []
    else:
        parts = getattr(response, "parts", None) or []

    seen_hashes = set()
    for part in parts:
        is_thought = bool(getattr(part, "thought", False))

        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None and getattr(inline_data, "data", None):
            image_bytes = inline_data.data
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            if image_hash not in seen_hashes:
                seen_hashes.add(image_hash)
                image_tensor = _image_part_to_tensor(image_bytes)
                image_artifact = ParsedGeminiImageArtifact(
                    mime_type=getattr(inline_data, "mime_type", None),
                    raw_bytes=image_bytes,
                    sha256=image_hash,
                    source="response_bytes",
                )
                if is_thought:
                    thinking_images.append(image_tensor)
                else:
                    normal_images.append(image_tensor)
                    normal_image_artifacts.append(image_artifact)

        text_value = getattr(part, "text", None)
        if not text_value:
            continue
        if is_thought:
            thought_chunks.append(text_value)
        else:
            answer_chunks.append(text_value)

    if not normal_images and thinking_images:
        normal_images.append(thinking_images[0])

    if normal_images:
        image_batch = torch.cat(normal_images, dim=0)
        no_image_messages: list[str] = []
    else:
        image_batch = blank_image_tensor()
        no_image_messages = list(NO_IMAGE_MESSAGES)

    if thinking_images:
        thinking_image_batch = torch.cat(thinking_images, dim=0)
    else:
        thinking_image_batch = blank_image_tensor()

    thinking_process = "\n".join(thought_chunks).strip() or THINKING_PROCESS_DEFAULT
    answer_text = "\n".join(answer_chunks).strip()

    if answer_text:
        system_messages = append_message_block(system_messages, answer_text)

    if not normal_images:
        if answer_text:
            no_image_messages.append("Text response was returned without an image.")
        no_image_messages.extend(_collect_response_diagnostics(response, first_candidate))
        system_messages = append_message_block(system_messages, "\n".join(no_image_messages))

    system_messages = system_messages.strip() or SYSTEM_MESSAGES_DEFAULT

    return ParsedGeminiResponse(
        image_batch=image_batch,
        thinking_image_batch=thinking_image_batch,
        thinking_process=thinking_process,
        system_messages=system_messages,
        image_artifacts=tuple(normal_image_artifacts),
    )


def append_message_block(system_messages: str, block_text: str) -> str:
    block_text = (block_text or "").strip()
    if not block_text:
        return system_messages
    if system_messages and system_messages.strip():
        return f"{system_messages.strip()}\n\n{block_text}"
    return block_text


def ensure_default_text(value: str, default: str) -> str:
    value = (value or "").strip()
    if value:
        return value
    return default


def _image_part_to_tensor(image_bytes: bytes) -> torch.Tensor:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    return torch.from_numpy(image_np).unsqueeze(0)


def _build_tools(selection: NormalizedModelSelection) -> list[Any] | None:
    search_mode = selection.values.get("search_mode")
    if search_mode in {None, "off"}:
        return None

    if selection.model_id == "gemini-3-pro-image-preview":
        return [types.Tool(google_search=types.GoogleSearch())]

    if selection.model_id == "gemini-3.1-flash-image-preview":
        search_type_kwargs = {}
        if search_mode in {"web", "web+image"}:
            search_type_kwargs["web_search"] = types.WebSearch()
        if search_mode in {"image", "web+image"}:
            search_type_kwargs["image_search"] = types.ImageSearch()
        google_search = types.GoogleSearch(
            search_types=types.SearchTypes(**search_type_kwargs)
        )
        return [types.Tool(google_search=google_search)]

    return None


def _build_thinking_config(selection: NormalizedModelSelection) -> Any | None:
    if selection.model_id != "gemini-3.1-flash-image-preview":
        return None

    thinking_level = selection.values.get("thinking_level")
    if thinking_level is None:
        return None

    return types.ThinkingConfig(
        thinking_level=_enum_value(types.ThinkingLevel, thinking_level),
        include_thoughts=selection.values.get("include_thoughts"),
    )


def _enum_value(enum_type: Any, member_name: str) -> Any:
    try:
        return getattr(enum_type, member_name)
    except AttributeError:
        return member_name


def _stringify_response_detail(value: Any) -> str:
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


def _collect_response_diagnostics(response: Any, candidate: Any) -> list[str]:
    diagnostics = []

    finish_reason = _stringify_response_detail(
        getattr(candidate, "finish_reason", None) if candidate is not None else None
    )
    if finish_reason:
        diagnostics.append(f"Finish reason: {finish_reason}")

    candidate_feedback = _stringify_response_detail(
        getattr(candidate, "safety_ratings", None) if candidate is not None else None
    )
    if candidate_feedback:
        diagnostics.append(f"Candidate feedback: {candidate_feedback}")

    prompt_feedback = _stringify_response_detail(getattr(response, "prompt_feedback", None))
    if prompt_feedback:
        diagnostics.append(f"Prompt feedback: {prompt_feedback}")

    return diagnostics
