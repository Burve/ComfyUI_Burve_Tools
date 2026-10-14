from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None


MIN_GOOGLE_GENAI_VERSION = "2.13.0"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-nano-banana-2.1"
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

GEMINI_REFERENCE_PIPE_KIND = "burve.gemini_reference_pipe"
GEMINI_REFERENCE_PIPE_VERSION = 1
GEMINI_INPUT_FILES_PIPE_KIND = "burve.gemini_input_files_pipe"
GEMINI_INPUT_FILES_PIPE_VERSION = 1
GEMINI_INTERACTION_PIPE_KIND = "burve.gemini_interaction_pipe"
GEMINI_INTERACTION_PIPE_VERSION = 1
GEMINI_CONTENT_PIPE_KIND = "burve.gemini_content_pipe"
GEMINI_CONTENT_PIPE_VERSION = 1
GROUNDING_RENDERER_VERSION = "burve-grounding-v1"

STANDARD_ASPECT_RATIOS = (
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
FLASH_31_ASPECT_RATIOS = (
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
)
SUPPORTED_DOCUMENT_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
}
SUPPORTED_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
}


@dataclass(frozen=True)
class GeminiImageFieldSpec:
    key: str
    field_type: str
    default: Any
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeminiImageModelSpec:
    model_id: str
    display_name: str
    aliases: tuple[str, ...]
    lifecycle_status: str
    retirement_date: str | None
    fields: tuple[GeminiImageFieldSpec, ...]
    providers: tuple[str, ...]
    api_modes: tuple[str, ...]
    vertex_global_only: bool
    input_modalities: tuple[str, ...]
    response_modes: tuple[str, ...]
    max_reference_images: int
    reference_role_limits: MappingProxyType
    search_modes: tuple[str, ...]
    thinking_levels: tuple[str, ...]
    output_mime_types: MappingProxyType
    documentation_url: str
    provider_api_modes: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    provider_aspect_ratios: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    provider_sampling_parameters: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class NormalizedModelSelection:
    model_id: str
    values: MappingProxyType
    requested_model_id: str = ""
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedGeminiRequest:
    provider_id: str
    requested_model_id: str
    model_id: str
    api_mode: str
    contents: list[Any]
    config: Any
    interaction_kwargs: MappingProxyType
    preflight_messages: tuple[str, ...]
    response_mode: str
    search_mode: str
    effective_location: str


class GeminiAspectRatioCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedGeminiImageArtifact:
    mime_type: str | None
    raw_bytes: bytes
    sha256: str
    source: str
    is_thought: bool = False
    sequence_index: int = 0
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ParsedGeminiResponse:
    image_batch: torch.Tensor
    thinking_image_batch: torch.Tensor
    thinking_process: str
    system_messages: str
    image_artifacts: tuple[ParsedGeminiImageArtifact, ...]
    response_text: str
    grounding_metadata_json: str
    content_pipe: dict[str, Any]
    interaction_id: str
    search_suggestions_html: str


def _combo_field(key: str, default: str, options: tuple[str, ...]) -> GeminiImageFieldSpec:
    return GeminiImageFieldSpec(key=key, field_type="combo", default=default, options=options)


def _bool_field(key: str, default: bool) -> GeminiImageFieldSpec:
    return GeminiImageFieldSpec(key=key, field_type="boolean", default=default)


def _role_limits(**values: int) -> MappingProxyType:
    return MappingProxyType(dict(values))


def _mime_types(**providers: tuple[str, ...]) -> MappingProxyType:
    return MappingProxyType(dict(providers))


GEMINI_IMAGE_MODEL_SPECS = MappingProxyType(
    {
        "gemini-nano-banana-2.1": GeminiImageModelSpec(
            model_id="gemini-nano-banana-2.1",
            display_name="Gemini Nano Banana 2.1",
            aliases=(),
            lifecycle_status="stable",
            retirement_date=None,
            fields=(
                _combo_field("aspect_ratio", "1:1", FLASH_31_ASPECT_RATIOS),
                _combo_field("resolution", "1K", ("1K", "2K", "4K")),
                _combo_field("search_mode", "off", ("off", "web", "image", "web+image")),
                _combo_field("thinking_level", "MEDIUM", ("MINIMAL", "MEDIUM", "HIGH")),
                _bool_field("include_thoughts", False),
                _combo_field("output_mime_type", "default", ("default", "image/jpeg")),
            ),
            providers=("aistudio", "vertex"),
            api_modes=("generate_content", "interactions"),
            vertex_global_only=True,
            input_modalities=("text", "image", "video", "document"),
            response_modes=("text_and_image", "image_only"),
            max_reference_images=14,
            reference_role_limits=_role_limits(object=10, character=4, style=0, generic=14),
            search_modes=("off", "web", "image", "web+image"),
            thinking_levels=("MINIMAL", "MEDIUM", "HIGH"),
            output_mime_types=_mime_types(
                aistudio=("image/png", "image/jpeg"),
                vertex=("image/png", "image/jpeg"),
            ),
            documentation_url="https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1",
            provider_api_modes=MappingProxyType({"vertex": ("generate_content",)}),
            provider_aspect_ratios=MappingProxyType({"vertex": FLASH_31_ASPECT_RATIOS + ("9:21",)}),
            provider_sampling_parameters=MappingProxyType({"vertex": ()}),
        ),
        "gemini-3.1-flash-image": GeminiImageModelSpec(
            model_id="gemini-3.1-flash-image",
            display_name="Gemini 3.1 Flash Image",
            aliases=("gemini-3.1-flash-image-preview",),
            lifecycle_status="deprecated",
            retirement_date=None,
            fields=(
                _combo_field("aspect_ratio", "1:1", FLASH_31_ASPECT_RATIOS),
                _combo_field("resolution", "1K", ("512", "1K", "2K", "4K")),
                _combo_field("search_mode", "off", ("off", "web", "image", "web+image")),
                _combo_field("thinking_level", "MINIMAL", ("MINIMAL", "HIGH")),
                _bool_field("include_thoughts", False),
                # Retained in the DynamicCombo so preview-era workflows deserialize.
                _combo_field(
                    "output_mime_type",
                    "image/png",
                    ("image/png", "image/jpeg", "image/webp"),
                ),
                _combo_field("prominent_people", "allow", ("allow", "block")),
            ),
            providers=("aistudio", "vertex"),
            api_modes=("generate_content", "interactions"),
            vertex_global_only=True,
            input_modalities=("text", "image", "video", "document"),
            response_modes=("text_and_image", "image_only"),
            max_reference_images=14,
            reference_role_limits=_role_limits(object=10, character=4, style=0, generic=14),
            search_modes=("off", "web", "image", "web+image"),
            thinking_levels=("MINIMAL", "HIGH"),
            output_mime_types=_mime_types(
                aistudio=("image/png", "image/jpeg"),
                vertex=("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"),
            ),
            documentation_url="https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image",
        ),
        "gemini-3-pro-image": GeminiImageModelSpec(
            model_id="gemini-3-pro-image",
            display_name="Gemini 3 Pro Image",
            aliases=("gemini-3-pro-image-preview",),
            lifecycle_status="stable",
            retirement_date=None,
            fields=(
                _combo_field("aspect_ratio", "1:1", STANDARD_ASPECT_RATIOS),
                _combo_field("resolution", "1K", ("1K", "2K", "4K")),
                _combo_field("search_mode", "off", ("off", "web")),
                _combo_field(
                    "output_mime_type",
                    "default",
                    ("default", "image/jpeg"),
                ),
            ),
            providers=("aistudio", "vertex"),
            api_modes=("generate_content", "interactions"),
            vertex_global_only=True,
            input_modalities=("text", "image", "document"),
            response_modes=("text_and_image", "image_only"),
            max_reference_images=14,
            reference_role_limits=_role_limits(object=6, character=5, style=3, generic=14),
            search_modes=("off", "web"),
            thinking_levels=(),
            output_mime_types=_mime_types(
                aistudio=("image/png", "image/jpeg"),
                vertex=("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"),
            ),
            documentation_url="https://ai.google.dev/gemini-api/docs/models/gemini-3-pro-image",
        ),
        "gemini-3.1-flash-lite-image": GeminiImageModelSpec(
            model_id="gemini-3.1-flash-lite-image",
            display_name="Gemini 3.1 Flash Lite Image",
            aliases=(),
            lifecycle_status="stable",
            retirement_date=None,
            fields=(
                _combo_field("aspect_ratio", "1:1", STANDARD_ASPECT_RATIOS),
                _combo_field("resolution", "1K", ("1K",)),
                _combo_field("thinking_level", "MINIMAL", ("MINIMAL", "HIGH")),
                _bool_field("include_thoughts", False),
            ),
            providers=("aistudio", "vertex"),
            api_modes=("generate_content",),
            vertex_global_only=True,
            input_modalities=("text", "image", "document"),
            response_modes=("text_and_image", "image_only"),
            max_reference_images=14,
            reference_role_limits=_role_limits(object=14, character=0, style=0, generic=14),
            search_modes=("off",),
            thinking_levels=("MINIMAL", "HIGH"),
            output_mime_types=_mime_types(
                aistudio=("image/png",),
                vertex=("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"),
            ),
            documentation_url="https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-image",
        ),
        "gemini-2.5-flash-image": GeminiImageModelSpec(
            model_id="gemini-2.5-flash-image",
            display_name="Gemini 2.5 Flash Image (legacy)",
            aliases=(),
            lifecycle_status="legacy",
            retirement_date="2027-03-15",
            fields=(
                _combo_field("aspect_ratio", "1:1", STANDARD_ASPECT_RATIOS),
            ),
            providers=("aistudio", "vertex"),
            api_modes=("generate_content",),
            vertex_global_only=False,
            input_modalities=("text", "image"),
            response_modes=("text_and_image", "image_only"),
            max_reference_images=3,
            reference_role_limits=_role_limits(object=3, character=3, style=3, generic=3),
            search_modes=("off",),
            thinking_levels=(),
            output_mime_types=_mime_types(
                aistudio=("image/png",),
                vertex=("image/png", "image/jpeg", "image/webp"),
            ),
            documentation_url="https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-image",
        ),
    }
)

GEMINI_IMAGE_MODEL_ALIASES = MappingProxyType(
    {
        alias: model_id
        for model_id, spec in GEMINI_IMAGE_MODEL_SPECS.items()
        for alias in spec.aliases
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
            "GenerateContentConfig",
            "ImageConfig",
            "ThinkingConfig",
            "ThinkingLevel",
            "SearchTypes",
            "WebSearch",
            "ImageSearch",
            "ProminentPeople",
        )
        if not hasattr(types, symbol)
    ]
    client_has_interactions = hasattr(genai.Client, "interactions")

    if parsed < (2, 13, 0) or parsed >= (3, 0, 0) or missing_symbols or not client_has_interactions:
        details = [f"detected google-genai version: {version}"]
        if missing_symbols:
            details.append(f"missing symbols: {', '.join(missing_symbols)}")
        if not client_has_interactions:
            details.append("missing client.interactions API")
        raise RuntimeError(
            "ComfyUI_Burve_Tools 3.0.0 requires google-genai>=2.13.0,<3.\n"
            + "\n".join(details)
        )


def get_model_spec(model_id: str) -> GeminiImageModelSpec:
    normalized_id = GEMINI_IMAGE_MODEL_ALIASES.get(model_id, model_id)
    try:
        return GEMINI_IMAGE_MODEL_SPECS[normalized_id]
    except KeyError as exc:
        supported = ", ".join(GEMINI_IMAGE_MODEL_SPECS)
        raise ValueError(f"Unsupported model: {model_id}. Supported models: {supported}.") from exc


def get_field_default(model_id: str, field_key: str) -> Any:
    spec = get_model_spec(model_id)
    for field in spec.fields:
        if field.key == field_key:
            return field.default
    raise KeyError(field_key)


def get_model_fields(
    model_id: str, provider_id: str = "aistudio",
) -> tuple[GeminiImageFieldSpec, ...]:
    spec = get_model_spec(model_id)
    if provider_id not in spec.providers:
        raise ValueError(f"{model_id} does not support Google provider '{provider_id}'.")
    ratios = spec.provider_aspect_ratios.get(provider_id)
    if ratios is None:
        return spec.fields
    return tuple(
        replace(model_field, options=ratios) if model_field.key == "aspect_ratio" else model_field
        for model_field in spec.fields
    )


def get_supported_api_modes(model_id: str, provider_id: str) -> tuple[str, ...]:
    spec = get_model_spec(model_id)
    if provider_id not in spec.providers:
        raise ValueError(f"{model_id} does not support Google provider '{provider_id}'.")
    return spec.provider_api_modes.get(provider_id, spec.api_modes)


def get_supported_sampling_parameters(model_id: str, provider_id: str) -> tuple[str, ...]:
    return get_model_spec(model_id).provider_sampling_parameters.get(
        provider_id, ("seed", "temperature", "top_p"),
    )


def build_dynamic_combo_options(io_module, provider_id: str = "aistudio") -> list[Any]:
    options = []
    for model_id, spec in GEMINI_IMAGE_MODEL_SPECS.items():
        if provider_id not in spec.providers:
            continue
        inputs = []
        for field in get_model_fields(model_id, provider_id):
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


def normalize_model_selection(model: Any, provider_id: str = "aistudio") -> NormalizedModelSelection:
    if isinstance(model, str):
        requested_model_id = model.strip()
        provided_values = {}
    elif isinstance(model, dict):
        requested_model_id = str(model.get("model") or "").strip()
        provided_values = {key: value for key, value in model.items() if key != "model"}
    else:
        raise ValueError("Model input must be a DynamicCombo payload or a model id string.")

    if not requested_model_id:
        raise ValueError("Model input is missing the selected model id.")

    model_id = GEMINI_IMAGE_MODEL_ALIASES.get(requested_model_id, requested_model_id)
    spec = get_model_spec(model_id)
    fields = get_model_fields(model_id, provider_id)
    messages: list[str] = []
    if model_id != requested_model_id:
        messages.append(
            f"Model migration: '{requested_model_id}' was shut down and has been "
            f"mapped to stable model '{model_id}'."
        )
    if spec.lifecycle_status == "legacy" and spec.retirement_date:
        messages.append(
            f"Lifecycle warning: {model_id} is legacy and is scheduled to retire on "
            f"{spec.retirement_date}. It is not automatically replaced."
        )
    if spec.lifecycle_status == "deprecated":
        messages.append(
            f"Lifecycle warning: {model_id} is deprecated; no shutdown date is announced. "
            f"Use {DEFAULT_GEMINI_IMAGE_MODEL} for new workflows. "
            "Your saved model selection is not automatically replaced."
        )

    allowed_keys = {field.key for field in fields}
    unexpected_keys = sorted(set(provided_values) - allowed_keys)
    if unexpected_keys:
        raise ValueError(
            f"Unsupported parameters for {requested_model_id}: {', '.join(unexpected_keys)}."
        )

    normalized_values: dict[str, Any] = {}
    for field in fields:
        value = provided_values.get(field.key, field.default)
        if field.field_type == "combo":
            if not isinstance(value, str):
                raise ValueError(f"{field.key} must be a string for {requested_model_id}.")
            if value not in field.options:
                raise ValueError(
                    f"Invalid value for {field.key} on {requested_model_id}. "
                    f"Expected one of: {', '.join(field.options)}."
                )
        elif field.field_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{field.key} must be a boolean for {requested_model_id}.")
        normalized_values[field.key] = value

    return NormalizedModelSelection(
        model_id=model_id,
        requested_model_id=requested_model_id,
        values=MappingProxyType(normalized_values),
        messages=tuple(messages),
    )


def resolve_model_selection_with_aspect_ratio_override(
    model: Any,
    aspect_ratio_override: Any = "",
    provider_id: str = "aistudio",
) -> NormalizedModelSelection:
    selection = normalize_model_selection(model, provider_id)
    if aspect_ratio_override is None:
        return selection
    if not isinstance(aspect_ratio_override, str):
        raise GeminiAspectRatioCompatibilityError("aspect_ratio_override must be a string.")

    normalized_override = aspect_ratio_override.strip()
    if not normalized_override:
        return selection

    fields = get_model_fields(selection.model_id, provider_id)
    aspect_ratio_field = next((field for field in fields if field.key == "aspect_ratio"), None)
    if aspect_ratio_field is None:
        raise GeminiAspectRatioCompatibilityError(
            f"Selected model {selection.model_id} does not support aspect_ratio "
            f"'{normalized_override}'."
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
        requested_model_id=selection.requested_model_id,
        values=MappingProxyType(resolved_values),
        messages=selection.messages,
    )


def resolve_api_mode(
    *,
    provider_id: str,
    model_id: str,
    requested_api_mode: str,
    conversation_mode: str,
) -> str:
    if provider_id not in {"aistudio", "vertex"}:
        raise ValueError(f"Unsupported Google provider: {provider_id}.")
    if requested_api_mode not in {"auto", "generate_content", "interactions"}:
        raise ValueError("api_mode must be auto, generate_content, or interactions.")
    if conversation_mode not in {"stateless", "stateful"}:
        raise ValueError("conversation_mode must be stateless or stateful.")

    supported_api_modes = get_supported_api_modes(model_id, provider_id)
    if requested_api_mode == "auto":
        api_mode = (
            "interactions"
            if provider_id == "aistudio" and "interactions" in supported_api_modes
            else "generate_content"
        )
    else:
        api_mode = requested_api_mode

    if api_mode not in supported_api_modes:
        explanation = (
            " Interactions support for this model on Vertex is unconfirmed."
            if provider_id == "vertex" and api_mode == "interactions"
            else ""
        )
        raise ValueError(
            f"{model_id} does not currently support api_mode '{api_mode}' on {provider_id}. "
            f"Supported API modes: {', '.join(supported_api_modes)}.{explanation}"
        )
    if conversation_mode == "stateful" and api_mode != "interactions":
        raise ValueError("Stateful conversation mode requires the Interactions API.")
    return api_mode


def resolve_effective_location(
    *,
    provider_id: str,
    model_id: str,
    configured_location: str,
    location_mode: str,
) -> str:
    if provider_id != "vertex":
        return ""
    if location_mode not in {"auto", "configured"}:
        raise ValueError("vertex_location_mode must be auto or configured.")

    configured_location = str(configured_location or "").strip()
    spec = get_model_spec(model_id)
    if spec.vertex_global_only:
        if location_mode == "configured" and configured_location.lower() != "global":
            raise ValueError(
                f"{model_id} is global-only on Vertex AI. Set GOOGLE_CLOUD_LOCATION=global "
                "or use vertex_location_mode=auto."
            )
        return "global"
    return configured_location


def prepare_gemini_request(
    *,
    provider_id: str,
    prompt: str,
    model: Any,
    seed: int,
    system_instructions: str = "",
    reference_images: Any = None,
    reference_pack: Any = None,
    input_files: Any = None,
    video: Any = None,
    video_url: str = "",
    aspect_ratio_override: Any = "",
    api_mode: str = "auto",
    response_mode: str = "text_and_image",
    conversation_mode: str = "stateless",
    previous_interaction: Any = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    configured_location: str = "",
    vertex_location_mode: str = "auto",
    vertex_output_mime_type: str = "default",
    output_compression_quality: int = 90,
    person_generation: str = "default",
    prominent_people: str = "default",
    grounding_renderer: str = "",
) -> PreparedGeminiRequest:
    selection = resolve_model_selection_with_aspect_ratio_override(
        model,
        aspect_ratio_override=aspect_ratio_override,
        provider_id=provider_id,
    )
    spec = get_model_spec(selection.model_id)
    preflight_messages = list(selection.messages)
    compatibility_api_mode = api_mode
    if (
        api_mode == "auto"
        and conversation_mode == "stateless"
        and selection.requested_model_id != selection.model_id
    ):
        compatibility_api_mode = "generate_content"
        preflight_messages.append(
            "Compatibility routing: a retired preview model selection keeps the "
            "GenerateContent request path while using its stable model ID."
        )
    resolved_api_mode = resolve_api_mode(
        provider_id=provider_id,
        model_id=selection.model_id,
        requested_api_mode=compatibility_api_mode,
        conversation_mode=conversation_mode,
    )
    effective_location = resolve_effective_location(
        provider_id=provider_id,
        model_id=selection.model_id,
        configured_location=configured_location,
        location_mode=vertex_location_mode,
    )

    if response_mode not in spec.response_modes:
        raise ValueError(
            f"{selection.model_id} does not support response_mode '{response_mode}'."
        )
    sampling_parameters = get_supported_sampling_parameters(selection.model_id, provider_id)
    if "temperature" in sampling_parameters and (
        not isinstance(temperature, (int, float)) or not 0.0 <= float(temperature) <= 2.0
    ):
        raise ValueError("temperature must be between 0.0 and 2.0.")
    if "top_p" in sampling_parameters and (
        not isinstance(top_p, (int, float)) or not 0.0 <= float(top_p) <= 1.0
    ):
        raise ValueError("top_p must be between 0.0 and 1.0.")
    ignored_sampling = [key for key in ("seed", "temperature", "top_p") if key not in sampling_parameters]
    if ignored_sampling:
        preflight_messages.append(
            f"{selection.model_id} on {provider_id} does not support "
            f"{', '.join(ignored_sampling)}; these controls are ignored and omitted from the request."
        )

    search_mode = str(selection.values.get("search_mode", "off"))
    if search_mode not in spec.search_modes:
        raise ValueError(f"{selection.model_id} does not support search_mode '{search_mode}'.")
    if search_mode in {"image", "web+image"} and grounding_renderer != GROUNDING_RENDERER_VERSION:
        raise ValueError(
            "Image Search requires the Burve grounding attribution renderer. "
            "Reload the ComfyUI frontend after updating this plugin."
        )

    references, reference_messages = _normalize_reference_inputs(
        reference_images=reference_images,
        reference_pack=reference_pack,
        spec=spec,
    )
    preflight_messages.extend(reference_messages)
    files = _normalize_input_files(input_files)
    if files and "document" not in spec.input_modalities:
        raise ValueError(f"{selection.model_id} does not support document input.")
    video_input = _normalize_video_input(video=video, video_url=video_url)
    if video_input is not None and "video" not in spec.input_modalities:
        raise ValueError(
            f"{selection.model_id} does not support video-to-image input. "
            f"Use {DEFAULT_GEMINI_IMAGE_MODEL}."
        )

    previous_interaction_id = _normalize_previous_interaction(
        previous_interaction=previous_interaction,
        provider_id=provider_id,
        model_id=selection.model_id,
        api_mode=resolved_api_mode,
    )
    if previous_interaction_id and conversation_mode != "stateful":
        raise ValueError(
            "previous_interaction is connected, but conversation_mode is stateless."
        )

    if provider_id == "vertex":
        if vertex_output_mime_type not in {
            "default",
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/heic",
            "image/heif",
        }:
            raise ValueError("Unsupported vertex_output_mime_type.")
        if vertex_output_mime_type != "default":
            allowed_mime_types = spec.output_mime_types.get("vertex", ())
            if vertex_output_mime_type not in allowed_mime_types:
                raise ValueError(
                    f"{selection.model_id} does not support Vertex output type "
                    f"{vertex_output_mime_type}."
                )
        if not isinstance(output_compression_quality, int) or not 0 <= output_compression_quality <= 100:
            raise ValueError("output_compression_quality must be between 0 and 100.")
        if person_generation not in {"default", "ALLOW_ALL", "ALLOW_ADULT", "ALLOW_NONE"}:
            raise ValueError("Unsupported person_generation value.")
        if prominent_people not in {"default", "allow", "block"}:
            raise ValueError("Unsupported prominent_people value.")

    if resolved_api_mode == "generate_content":
        config = _build_generate_content_config(
            provider_id=provider_id,
            selection=selection,
            seed=seed,
            system_instructions=system_instructions,
            response_mode=response_mode,
            temperature=float(temperature) if "temperature" in sampling_parameters else None,
            top_p=float(top_p) if "top_p" in sampling_parameters else None,
            vertex_output_mime_type=vertex_output_mime_type,
            output_compression_quality=output_compression_quality,
            person_generation=person_generation,
            prominent_people=prominent_people,
            preflight_messages=preflight_messages,
        )
        contents = _build_generate_content_contents(
            prompt=prompt,
            references=references,
            files=files,
            video_input=video_input,
        )
        interaction_kwargs: dict[str, Any] = {}
    else:
        contents = []
        config = None
        interaction_kwargs = _build_interaction_kwargs(
            provider_id=provider_id,
            selection=selection,
            prompt=prompt,
            references=references,
            files=files,
            video_input=video_input,
            seed=seed,
            system_instructions=system_instructions,
            response_mode=response_mode,
            conversation_mode=conversation_mode,
            previous_interaction_id=previous_interaction_id,
            temperature=float(temperature) if "temperature" in sampling_parameters else None,
            top_p=float(top_p) if "top_p" in sampling_parameters else None,
            vertex_output_mime_type=vertex_output_mime_type,
        )

    if provider_id == "vertex" and effective_location:
        preflight_messages.append(
            f"Request routing: provider=vertex, api_mode={resolved_api_mode}, "
            f"effective_location={effective_location}."
        )
    else:
        preflight_messages.append(
            f"Request routing: provider={provider_id}, api_mode={resolved_api_mode}."
        )

    return PreparedGeminiRequest(
        provider_id=provider_id,
        requested_model_id=selection.requested_model_id,
        model_id=selection.model_id,
        api_mode=resolved_api_mode,
        contents=contents,
        config=config,
        interaction_kwargs=MappingProxyType(interaction_kwargs),
        preflight_messages=tuple(preflight_messages),
        response_mode=response_mode,
        search_mode=search_mode,
        effective_location=effective_location,
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
    **kwargs: Any,
) -> PreparedGeminiRequest:
    """Backward-compatible request helper retained for third-party imports and tests."""
    return prepare_gemini_request(
        provider_id=provider_id,
        prompt=prompt,
        model=model,
        seed=seed,
        system_instructions=system_instructions,
        reference_images=reference_images,
        aspect_ratio_override=aspect_ratio_override,
        api_mode="generate_content",
        **kwargs,
    )


def clone_config_with_http_timeout(config: Any, timeout_ms: int) -> Any:
    if hasattr(config, "model_dump"):
        config_kwargs = config.model_dump(exclude_none=False)
    else:
        config_kwargs = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    config_kwargs["http_options"] = types.HttpOptions(timeout=int(timeout_ms))
    return types.GenerateContentConfig(**config_kwargs)


def build_contents(*, prompt: str, reference_images: Any = None) -> list[Any]:
    """Compatibility wrapper for the legacy public helper."""
    references, _ = _normalize_reference_inputs(
        reference_images=reference_images,
        reference_pack=None,
        spec=get_model_spec("gemini-3.1-flash-image"),
    )
    return _build_generate_content_contents(
        prompt=prompt,
        references=references,
        files=(),
        video_input=None,
    )


def parse_generate_content_response(
    response: Any,
    preflight_messages: tuple[str, ...] = (),
) -> ParsedGeminiResponse:
    normal_images: list[torch.Tensor] = []
    thinking_images: list[torch.Tensor] = []
    artifacts: list[ParsedGeminiImageArtifact] = []
    thought_chunks: list[str] = []
    answer_chunks: list[str] = []
    blocks: list[dict[str, Any]] = []
    parse_diagnostics: list[str] = []
    system_messages = "\n".join(message for message in preflight_messages if message).strip()

    candidates = getattr(response, "candidates", None)
    parts: list[Any] = []
    first_candidate = None
    if candidates:
        first_candidate = candidates[0]
        content = getattr(first_candidate, "content", None)
        if content is not None:
            parts = list(getattr(content, "parts", None) or [])
    else:
        parts = list(getattr(response, "parts", None) or [])

    normal_seen: set[str] = set()
    thought_seen: set[str] = set()
    for part in parts:
        is_thought = bool(getattr(part, "thought", False))
        inline_data = getattr(part, "inline_data", None)
        raw_data = getattr(inline_data, "data", None) if inline_data is not None else None
        if raw_data:
            try:
                image_bytes = _decode_image_data(raw_data)
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                seen = thought_seen if is_thought else normal_seen
                if image_hash not in seen:
                    seen.add(image_hash)
                    image_tensor, width, height = _image_part_to_tensor(image_bytes)
                    if is_thought:
                        thinking_images.append(image_tensor)
                    else:
                        normal_images.append(image_tensor)
                    artifact_index = len(artifacts)
                    artifacts.append(
                        ParsedGeminiImageArtifact(
                            mime_type=getattr(inline_data, "mime_type", None),
                            raw_bytes=image_bytes,
                            sha256=image_hash,
                            source="response_bytes",
                            is_thought=is_thought,
                            sequence_index=len(blocks),
                            width=width,
                            height=height,
                        )
                    )
                    blocks.append(
                        {
                            "type": "image",
                            "artifact_index": artifact_index,
                            "is_thought": is_thought,
                        }
                    )
            except Exception as exc:
                parse_diagnostics.append(
                    "Malformed GenerateContent image part was skipped: "
                    f"{type(exc).__name__}: {exc}"
                )
                blocks.append(
                    {
                        "type": "malformed_image",
                        "is_thought": is_thought,
                        "error": str(exc),
                    }
                )

        text_value = getattr(part, "text", None)
        if text_value:
            target = thought_chunks if is_thought else answer_chunks
            target.append(str(text_value))
            blocks.append(
                {
                    "type": "text",
                    "text": str(text_value),
                    "is_thought": is_thought,
                    "annotations": _serialize_annotations(getattr(part, "annotations", None)),
                }
            )

    image_batch, omitted_normal = _compatible_batch(normal_images)
    thinking_image_batch, omitted_thinking = _compatible_batch(thinking_images)
    no_image_messages: list[str] = []
    if not normal_images:
        if thinking_images:
            image_batch = thinking_images[0]
            no_image_messages.append(
                "No final image was returned; the first thought image is exposed on the legacy image output."
            )
        else:
            image_batch = blank_image_tensor()
            no_image_messages.extend(NO_IMAGE_MESSAGES)
    if not thinking_images:
        thinking_image_batch = blank_image_tensor()

    if omitted_normal:
        no_image_messages.append(
            f"{omitted_normal} differently sized final image(s) remain available in "
            "generated_image_pipe/interleaved_content but were omitted from the legacy batch."
        )
    if omitted_thinking:
        no_image_messages.append(
            f"{omitted_thinking} differently sized thought image(s) remain available in "
            "generated_image_pipe/interleaved_content but were omitted from the legacy batch."
        )
    no_image_messages.extend(parse_diagnostics)

    thinking_process = "\n".join(thought_chunks).strip() or THINKING_PROCESS_DEFAULT
    response_text = "\n".join(answer_chunks).strip()
    if response_text:
        system_messages = append_message_block(system_messages, response_text)
    if not normal_images:
        if response_text:
            no_image_messages.append("Text response was returned without an image.")
        no_image_messages.extend(_collect_response_diagnostics(response, first_candidate))
    if no_image_messages:
        system_messages = append_message_block(system_messages, "\n".join(no_image_messages))

    grounding, suggestions_html = _extract_generate_content_grounding(
        response=response,
        candidate=first_candidate,
    )
    content_pipe = _build_content_pipe(blocks=blocks, artifacts=artifacts)
    return ParsedGeminiResponse(
        image_batch=image_batch,
        thinking_image_batch=thinking_image_batch,
        thinking_process=thinking_process,
        system_messages=system_messages.strip() or SYSTEM_MESSAGES_DEFAULT,
        image_artifacts=tuple(artifacts),
        response_text=response_text,
        grounding_metadata_json=json.dumps(grounding, ensure_ascii=False, indent=2),
        content_pipe=content_pipe,
        interaction_id="",
        search_suggestions_html=suggestions_html,
    )


def parse_interaction_response(
    interaction: Any,
    preflight_messages: tuple[str, ...] = (),
) -> ParsedGeminiResponse:
    normal_images: list[torch.Tensor] = []
    thinking_images: list[torch.Tensor] = []
    artifacts: list[ParsedGeminiImageArtifact] = []
    thought_chunks: list[str] = []
    answer_chunks: list[str] = []
    blocks: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    search_calls: list[dict[str, Any]] = []
    search_suggestions: list[str] = []
    parse_diagnostics: list[str] = []
    normal_seen: set[str] = set()
    thought_seen: set[str] = set()
    system_messages = "\n".join(message for message in preflight_messages if message).strip()

    for step in list(getattr(interaction, "steps", None) or []):
        step_type = str(getattr(step, "type", "") or "")
        if step_type == "thought":
            for item in list(getattr(step, "summary", None) or []):
                _consume_interaction_content(
                    item=item,
                    is_thought=True,
                    normal_images=normal_images,
                    thinking_images=thinking_images,
                    artifacts=artifacts,
                    normal_seen=normal_seen,
                    thought_seen=thought_seen,
                    thought_chunks=thought_chunks,
                    answer_chunks=answer_chunks,
                    blocks=blocks,
                    citations=citations,
                    diagnostics=parse_diagnostics,
                )
        elif step_type == "model_output":
            for item in list(getattr(step, "content", None) or []):
                _consume_interaction_content(
                    item=item,
                    is_thought=False,
                    normal_images=normal_images,
                    thinking_images=thinking_images,
                    artifacts=artifacts,
                    normal_seen=normal_seen,
                    thought_seen=thought_seen,
                    thought_chunks=thought_chunks,
                    answer_chunks=answer_chunks,
                    blocks=blocks,
                    citations=citations,
                    diagnostics=parse_diagnostics,
                )
            step_error = getattr(step, "error", None)
            if step_error:
                system_messages = append_message_block(
                    system_messages,
                    f"Interaction model-output error: {_stringify_response_detail(step_error)}",
                )
        elif step_type == "google_search_call":
            search_calls.append(
                {
                    "id": getattr(step, "id", None),
                    "search_type": getattr(step, "search_type", None),
                    "arguments": _to_jsonable(getattr(step, "arguments", None)),
                }
            )
        elif step_type == "google_search_result":
            for result in list(getattr(step, "result", None) or []):
                html = getattr(result, "search_suggestions", None)
                if isinstance(html, str) and html.strip():
                    search_suggestions.append(html)
        elif step_type:
            blocks.append({"type": "step", "step_type": step_type})

    output_image = getattr(interaction, "output_image", None)
    if not normal_images and output_image is not None:
        _consume_interaction_content(
            item=output_image,
            is_thought=False,
            normal_images=normal_images,
            thinking_images=thinking_images,
            artifacts=artifacts,
            normal_seen=normal_seen,
            thought_seen=thought_seen,
            thought_chunks=thought_chunks,
            answer_chunks=answer_chunks,
            blocks=blocks,
            citations=citations,
            diagnostics=parse_diagnostics,
        )

    image_batch, omitted_normal = _compatible_batch(normal_images)
    thinking_image_batch, omitted_thinking = _compatible_batch(thinking_images)
    messages: list[str] = []
    if not normal_images:
        if thinking_images:
            image_batch = thinking_images[0]
            messages.append(
                "No final image was returned; the first thought image is exposed on the legacy image output."
            )
        else:
            image_batch = blank_image_tensor()
            messages.extend(NO_IMAGE_MESSAGES)
    if not thinking_images:
        thinking_image_batch = blank_image_tensor()
    if omitted_normal:
        messages.append(
            f"{omitted_normal} differently sized final image(s) were omitted from the legacy batch."
        )
    if omitted_thinking:
        messages.append(
            f"{omitted_thinking} differently sized thought image(s) were omitted from the legacy batch."
        )
    messages.extend(parse_diagnostics)

    response_text = "\n".join(answer_chunks).strip()
    if not response_text:
        output_text = getattr(interaction, "output_text", None)
        if isinstance(output_text, str):
            response_text = output_text.strip()
    if response_text:
        system_messages = append_message_block(system_messages, response_text)
    if messages:
        system_messages = append_message_block(system_messages, "\n".join(messages))

    usage = _to_jsonable(getattr(interaction, "usage", None))
    grounding = {
        "api_mode": "interactions",
        "search_calls": search_calls,
        "citations": citations,
        "search_suggestions_html": search_suggestions,
        "usage": usage or {},
    }
    interaction_id = str(getattr(interaction, "id", "") or "")
    content_pipe = _build_content_pipe(blocks=blocks, artifacts=artifacts)
    return ParsedGeminiResponse(
        image_batch=image_batch,
        thinking_image_batch=thinking_image_batch,
        thinking_process="\n".join(thought_chunks).strip() or THINKING_PROCESS_DEFAULT,
        system_messages=system_messages.strip() or SYSTEM_MESSAGES_DEFAULT,
        image_artifacts=tuple(artifacts),
        response_text=response_text,
        grounding_metadata_json=json.dumps(grounding, ensure_ascii=False, indent=2),
        content_pipe=content_pipe,
        interaction_id=interaction_id,
        search_suggestions_html="\n".join(search_suggestions),
    )


def build_interaction_pipe(
    *,
    provider_id: str,
    model_id: str,
    api_mode: str,
    interaction_id: str,
    stored: bool,
) -> dict[str, Any]:
    return {
        "kind": GEMINI_INTERACTION_PIPE_KIND,
        "version": GEMINI_INTERACTION_PIPE_VERSION,
        "provider_id": provider_id,
        "model_id": model_id,
        "api_mode": api_mode,
        "interaction_id": interaction_id,
        "stored": bool(stored),
    }


def append_message_block(system_messages: str, block_text: str) -> str:
    block_text = (block_text or "").strip()
    if not block_text:
        return system_messages
    if system_messages and system_messages.strip():
        return f"{system_messages.strip()}\n\n{block_text}"
    return block_text


def ensure_default_text(value: str, default: str) -> str:
    value = (value or "").strip()
    return value if value else default


def _build_sampling_config(
    *, provider_id: str, model_id: str, seed: int, temperature: float | None, top_p: float | None,
) -> dict[str, Any]:
    supported = get_supported_sampling_parameters(model_id, provider_id)
    config = {}
    if "seed" in supported and seed is not None:
        config["seed"] = seed % 2147483647
    if "temperature" in supported and temperature is not None:
        config["temperature"] = temperature
    if "top_p" in supported and top_p is not None:
        config["top_p"] = top_p
    return config


def _build_generate_content_config(
    *,
    provider_id: str,
    selection: NormalizedModelSelection,
    seed: int,
    system_instructions: str,
    response_mode: str,
    temperature: float | None,
    top_p: float | None,
    vertex_output_mime_type: str,
    output_compression_quality: int,
    person_generation: str,
    prominent_people: str,
    preflight_messages: list[str],
) -> Any:
    image_config_kwargs: dict[str, Any] = {
        "aspect_ratio": selection.values["aspect_ratio"],
    }
    if "resolution" in selection.values:
        image_config_kwargs["image_size"] = selection.values["resolution"]

    if provider_id == "vertex":
        legacy_mime = selection.values.get("output_mime_type")
        if legacy_mime == "default":
            legacy_mime = None
        selected_mime = (
            vertex_output_mime_type
            if vertex_output_mime_type != "default"
            else legacy_mime
        )
        if selected_mime:
            if selected_mime not in get_model_spec(selection.model_id).output_mime_types["vertex"]:
                raise ValueError(
                    f"{selection.model_id} does not support Vertex output type {selected_mime}."
                )
            image_config_kwargs["output_mime_type"] = selected_mime
            if selected_mime == "image/jpeg":
                image_config_kwargs["output_compression_quality"] = output_compression_quality
        if person_generation != "default":
            image_config_kwargs["person_generation"] = person_generation

        selected_prominent_people = prominent_people
        if selected_prominent_people == "default":
            selected_prominent_people = selection.values.get("prominent_people", "default")
        if selected_prominent_people in {"allow", "block"}:
            image_config_kwargs["prominent_people"] = _enum_value(
                types.ProminentPeople,
                {
                    "allow": "ALLOW_PROMINENT_PEOPLE",
                    "block": "BLOCK_PROMINENT_PEOPLE",
                }[selected_prominent_people],
            )
        if person_generation == "ALLOW_NONE" and selected_prominent_people == "allow":
            preflight_messages.append(
                "person_generation=ALLOW_NONE takes precedence over prominent_people=allow."
            )
    elif "output_mime_type" in selection.values:
        selected = selection.values["output_mime_type"]
        if selected != get_field_default(selection.model_id, "output_mime_type"):
            preflight_messages.append(
                "DynamicCombo output_mime_type is ignored by the Gemini GenerateContent API."
            )
        if (
            "prominent_people" in selection.values
            and selection.values.get("prominent_people")
            != get_field_default(selection.model_id, "prominent_people")
        ):
            preflight_messages.append(
                "DynamicCombo prominent_people is ignored by the Gemini API."
            )

    config_kwargs: dict[str, Any] = {
        "response_modalities": ["IMAGE"]
        if response_mode == "image_only"
        else ["TEXT", "IMAGE"],
        "image_config": types.ImageConfig(**image_config_kwargs),
        "system_instruction": system_instructions or None,
    }
    config_kwargs.update(_build_sampling_config(
        provider_id=provider_id, model_id=selection.model_id,
        seed=seed, temperature=temperature, top_p=top_p,
    ))
    tools = _build_generate_content_tools(selection)
    if tools:
        config_kwargs["tools"] = tools
    thinking_config = _build_generate_content_thinking_config(selection)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    return types.GenerateContentConfig(**config_kwargs)


def _build_interaction_kwargs(
    *,
    provider_id: str,
    selection: NormalizedModelSelection,
    prompt: str,
    references: tuple[dict[str, Any], ...],
    files: tuple[dict[str, Any], ...],
    video_input: dict[str, Any] | None,
    seed: int,
    system_instructions: str,
    response_mode: str,
    conversation_mode: str,
    previous_interaction_id: str,
    temperature: float | None,
    top_p: float | None,
    vertex_output_mime_type: str,
) -> dict[str, Any]:
    input_blocks: list[dict[str, Any]] = []
    for reference in references:
        reference_instruction = _build_reference_instruction(reference)
        if reference_instruction:
            input_blocks.append({"type": "text", "text": reference_instruction})
        image_bytes, mime_type = _tensor_to_png_bytes(reference["image"])
        input_blocks.append(
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode("ascii"),
                "mime_type": mime_type,
            }
        )
    for file_item in files:
        input_blocks.append(
            {
                "type": "document",
                "data": base64.b64encode(file_item["data"]).decode("ascii"),
                "mime_type": file_item["mime_type"],
            }
        )
    if video_input is not None:
        video_block = {
            "type": "video",
            "mime_type": video_input["mime_type"],
        }
        if video_input.get("uri"):
            video_block["uri"] = video_input["uri"]
        else:
            video_block["data"] = base64.b64encode(video_input["data"]).decode("ascii")
        input_blocks.append(video_block)
    input_blocks.append({"type": "text", "text": prompt})

    image_format: dict[str, Any] = {
        "type": "image",
        "aspect_ratio": selection.values["aspect_ratio"],
    }
    if "resolution" in selection.values:
        image_format["image_size"] = selection.values["resolution"]
    if provider_id == "aistudio" and "output_mime_type" in selection.values:
        selected_mime = selection.values["output_mime_type"]
        # The 2.13 Interactions contract exposes JPEG as the only explicit MIME
        # value. Omitting the field preserves Google's default image encoding.
        if selected_mime == "image/jpeg":
            image_format["mime_type"] = selected_mime
    elif provider_id == "vertex" and vertex_output_mime_type != "default":
        image_format["mime_type"] = vertex_output_mime_type

    response_format: Any = image_format
    if response_mode == "text_and_image":
        response_format = [{"type": "text"}, image_format]

    generation_config = _build_sampling_config(
        provider_id=provider_id, model_id=selection.model_id,
        seed=seed, temperature=temperature, top_p=top_p,
    )
    thinking_level = selection.values.get("thinking_level")
    if thinking_level:
        generation_config["thinking_level"] = str(thinking_level).lower()
        generation_config["thinking_summaries"] = (
            "auto" if selection.values.get("include_thoughts") else "none"
        )

    kwargs: dict[str, Any] = {
        "model": selection.model_id,
        "input": input_blocks,
        "store": conversation_mode == "stateful",
        "system_instruction": system_instructions or None,
        "response_format": response_format,
        "generation_config": {key: value for key, value in generation_config.items() if value is not None},
    }
    tools = _build_interaction_tools(selection)
    if tools:
        kwargs["tools"] = tools
    if previous_interaction_id:
        kwargs["previous_interaction_id"] = previous_interaction_id
    return {key: value for key, value in kwargs.items() if value is not None}


def _build_generate_content_contents(
    *,
    prompt: str,
    references: tuple[dict[str, Any], ...],
    files: tuple[dict[str, Any], ...],
    video_input: dict[str, Any] | None,
) -> list[Any]:
    contents: list[Any] = [prompt]
    for reference in references:
        reference_instruction = _build_reference_instruction(reference)
        if reference_instruction:
            contents.append(reference_instruction)
        contents.append(_tensor_to_pil(reference["image"]))
    for file_item in files:
        contents.append(
            types.Part.from_bytes(
                data=file_item["data"],
                mime_type=file_item["mime_type"],
            )
        )
    if video_input is not None:
        if video_input.get("uri"):
            contents.append(
                types.Part.from_uri(
                    file_uri=video_input["uri"],
                    mime_type=video_input["mime_type"],
                )
            )
        else:
            contents.append(
                types.Part.from_bytes(
                    data=video_input["data"],
                    mime_type=video_input["mime_type"],
                )
            )
    return contents


def _build_reference_instruction(reference: dict[str, Any]) -> str:
    role = str(reference.get("role", "generic") or "generic")
    label = str(reference.get("label", "") or "").strip()
    if role == "generic" and not label:
        return ""
    instruction = f"The next image is a {role} reference."
    if label:
        instruction += f" Reference label: {label}."
    return instruction


def _normalize_reference_inputs(
    *,
    reference_images: Any,
    reference_pack: Any,
    spec: GeminiImageModelSpec,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    messages: list[str] = []
    items: list[dict[str, Any]] = []
    if reference_pack is not None:
        if not isinstance(reference_pack, dict):
            raise ValueError("reference_pack must be a Gemini Reference Pack payload.")
        if reference_pack.get("kind") != GEMINI_REFERENCE_PIPE_KIND:
            raise ValueError("Invalid reference_pack kind.")
        if reference_pack.get("version") != GEMINI_REFERENCE_PIPE_VERSION:
            raise ValueError("Unsupported reference_pack version.")
        raw_items = reference_pack.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("reference_pack items must be a list.")
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"reference_pack item {index} must be a dict.")
            tensor = item.get("image")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"reference_pack item {index} image must be an IMAGE tensor.")
            role = str(item.get("role", "generic") or "generic")
            if role not in {"generic", "object", "character", "style"}:
                raise ValueError(f"reference_pack item {index} has unsupported role '{role}'.")
            for frame_index in range(tensor.shape[0]):
                items.append(
                    {
                        "image": tensor[frame_index : frame_index + 1],
                        "role": role,
                        "label": str(item.get("label", "") or ""),
                    }
                )
        if reference_images is not None:
            messages.append(
                "reference_pack is connected; legacy reference_images were ignored."
            )
    elif reference_images is not None:
        try:
            for tensor in reference_images:
                if not isinstance(tensor, torch.Tensor):
                    continue
                for frame_index in range(tensor.shape[0]):
                    items.append(
                        {
                            "image": tensor[frame_index : frame_index + 1],
                            "role": "generic",
                            "label": "",
                        }
                    )
        except TypeError as exc:
            raise ValueError("reference_images must be an IMAGE_LIST.") from exc

    if len(items) > spec.max_reference_images:
        raise ValueError(
            f"{spec.model_id} accepts at most {spec.max_reference_images} reference image(s); "
            f"{len(items)} were supplied."
        )
    role_counts: dict[str, int] = {}
    for item in items:
        role = item["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    for role, count in role_counts.items():
        limit = spec.reference_role_limits.get(role, spec.max_reference_images)
        if count > limit:
            raise ValueError(
                f"{spec.model_id} accepts at most {limit} '{role}' reference image(s); "
                f"{count} were supplied."
            )
    return tuple(items), tuple(messages)


def _normalize_input_files(input_files: Any) -> tuple[dict[str, Any], ...]:
    if input_files is None:
        return ()
    if not isinstance(input_files, dict):
        raise ValueError("input_files must be a Gemini Input Files payload.")
    if input_files.get("kind") != GEMINI_INPUT_FILES_PIPE_KIND:
        raise ValueError("Invalid input_files kind.")
    if input_files.get("version") != GEMINI_INPUT_FILES_PIPE_VERSION:
        raise ValueError("Unsupported input_files version.")
    raw_items = input_files.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("input_files items must be a list.")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"input_files item {index} must be a dict.")
        raw_data = item.get("data")
        path_value = item.get("path")
        if raw_data is not None:
            if not isinstance(raw_data, (bytes, bytearray)):
                raise ValueError(f"input_files item {index} data must be bytes.")
            data = bytes(raw_data)
            suffix = str(item.get("suffix", "") or "").lower()
        elif path_value:
            path = Path(str(path_value)).resolve()
            if not path.is_file():
                raise ValueError(f"input_files item {index} does not exist: {path}.")
            suffix = path.suffix.lower()
            data = path.read_bytes()
        else:
            raise ValueError(f"input_files item {index} has no path or data.")
        mime_type = str(item.get("mime_type") or SUPPORTED_DOCUMENT_MIME_TYPES.get(suffix, ""))
        if mime_type not in set(SUPPORTED_DOCUMENT_MIME_TYPES.values()):
            raise ValueError(
                f"input_files item {index} has unsupported document MIME type '{mime_type}'."
            )
        items.append(
            {
                "data": data,
                "mime_type": mime_type,
                "name": str(item.get("name", "") or ""),
            }
        )
    return tuple(items)


def _normalize_video_input(*, video: Any, video_url: str) -> dict[str, Any] | None:
    video_url = str(video_url or "").strip()
    if video is not None and video_url:
        raise ValueError("Connect either video or video_url, not both.")
    if video_url:
        parsed = urlparse(video_url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or hostname not in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
        }:
            raise ValueError("video_url must be a public HTTPS YouTube URL.")
        return {"uri": video_url, "mime_type": "video/mp4"}
    if video is None:
        return None

    source = video
    if hasattr(video, "get_stream_source"):
        source = video.get_stream_source()
    if isinstance(source, io.BytesIO):
        source.seek(0)
        data = source.read()
        mime_type = "video/mp4"
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        mime_type = "video/mp4"
    elif isinstance(source, str):
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(f"Video input file does not exist: {path}.")
        mime_type = SUPPORTED_VIDEO_MIME_TYPES.get(path.suffix.lower())
        if not mime_type:
            raise ValueError(f"Unsupported video file type: {path.suffix}.")
        data = path.read_bytes()
    else:
        raise ValueError("Unsupported VIDEO input object.")
    return {"data": data, "mime_type": mime_type}


def _normalize_previous_interaction(
    *,
    previous_interaction: Any,
    provider_id: str,
    model_id: str,
    api_mode: str,
) -> str:
    if previous_interaction is None:
        return ""
    if not isinstance(previous_interaction, dict):
        raise ValueError("previous_interaction must be a Gemini Interaction Pipe.")
    if previous_interaction.get("kind") != GEMINI_INTERACTION_PIPE_KIND:
        raise ValueError("Invalid previous_interaction kind.")
    if previous_interaction.get("version") != GEMINI_INTERACTION_PIPE_VERSION:
        raise ValueError("Unsupported previous_interaction version.")
    if not bool(previous_interaction.get("stored", False)):
        raise ValueError(
            "previous_interaction was created with store=false and cannot be resumed. "
            "Create the earlier turn with conversation_mode=stateful."
        )
    expected = {
        "provider_id": provider_id,
        "model_id": model_id,
        "api_mode": api_mode,
    }
    for key, expected_value in expected.items():
        actual = str(previous_interaction.get(key, "") or "")
        if actual != expected_value:
            raise ValueError(
                f"previous_interaction {key} mismatch: expected '{expected_value}', got '{actual}'."
            )
    interaction_id = str(previous_interaction.get("interaction_id", "") or "").strip()
    if not interaction_id:
        raise ValueError("previous_interaction is missing interaction_id.")
    return interaction_id


def _build_generate_content_tools(selection: NormalizedModelSelection) -> list[Any] | None:
    search_mode = selection.values.get("search_mode")
    if search_mode in {None, "off"}:
        return None
    if "image" in get_model_spec(selection.model_id).search_modes:
        search_type_kwargs: dict[str, Any] = {}
        if search_mode in {"web", "web+image"}:
            search_type_kwargs["web_search"] = types.WebSearch()
        if search_mode in {"image", "web+image"}:
            search_type_kwargs["image_search"] = types.ImageSearch()
        return [
            types.Tool(
                google_search=types.GoogleSearch(
                    search_types=types.SearchTypes(**search_type_kwargs)
                )
            )
        ]
    return [types.Tool(google_search=types.GoogleSearch())]


def _build_interaction_tools(selection: NormalizedModelSelection) -> list[dict[str, Any]] | None:
    search_mode = selection.values.get("search_mode")
    if search_mode in {None, "off"}:
        return None
    search_types = []
    if search_mode in {"web", "web+image"}:
        search_types.append("web_search")
    if search_mode in {"image", "web+image"}:
        search_types.append("image_search")
    tool: dict[str, Any] = {"type": "google_search"}
    if search_types:
        tool["search_types"] = search_types
    return [tool]


def _build_generate_content_thinking_config(
    selection: NormalizedModelSelection,
) -> Any | None:
    thinking_level = selection.values.get("thinking_level")
    if thinking_level is None:
        return None
    return types.ThinkingConfig(
        thinking_level=_enum_value(types.ThinkingLevel, str(thinking_level)),
        include_thoughts=bool(selection.values.get("include_thoughts")),
    )


def _enum_value(enum_type: Any, member_name: str) -> Any:
    try:
        return getattr(enum_type, member_name)
    except AttributeError:
        return member_name


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    if image_tensor.ndim != 4 or image_tensor.shape[0] < 1:
        raise ValueError("Reference IMAGE tensors must have shape [N,H,W,C].")
    frame = image_tensor[0]
    image_np = np.clip(frame.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np).convert("RGB")


def _tensor_to_png_bytes(image_tensor: torch.Tensor) -> tuple[bytes, str]:
    output = io.BytesIO()
    _tensor_to_pil(image_tensor).save(output, format="PNG")
    return output.getvalue(), "image/png"


def _decode_image_data(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except Exception as exc:
            raise ValueError("Interaction image data was not valid base64.") from exc
    if hasattr(data, "read"):
        return data.read()
    raise ValueError(f"Unsupported response image data type: {type(data).__name__}.")


def _image_part_to_tensor(
    image_bytes: bytes,
) -> tuple[torch.Tensor, int | None, int | None]:
    opened = Image.open(io.BytesIO(image_bytes))
    pil_image = opened.convert("RGB")
    size = getattr(pil_image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        width, height = int(size[0]), int(size[1])
    else:
        width = height = None
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    close = getattr(opened, "close", None)
    if callable(close):
        close()
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    return torch.from_numpy(image_np).unsqueeze(0), width, height


def _compatible_batch(images: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
    if not images:
        return blank_image_tensor(), 0
    expected_shape = tuple(images[0].shape[1:])
    compatible = [image for image in images if tuple(image.shape[1:]) == expected_shape]
    return torch.cat(compatible, dim=0), len(images) - len(compatible)


def _consume_interaction_content(
    *,
    item: Any,
    is_thought: bool,
    normal_images: list[torch.Tensor],
    thinking_images: list[torch.Tensor],
    artifacts: list[ParsedGeminiImageArtifact],
    normal_seen: set[str],
    thought_seen: set[str],
    thought_chunks: list[str],
    answer_chunks: list[str],
    blocks: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    diagnostics: list[str],
) -> None:
    content_type = str(getattr(item, "type", "") or "")
    if content_type == "text":
        text_value = str(getattr(item, "text", "") or "")
        if text_value:
            (thought_chunks if is_thought else answer_chunks).append(text_value)
            serialized_annotations = _serialize_annotations(getattr(item, "annotations", None))
            citations.extend(serialized_annotations)
            blocks.append(
                {
                    "type": "text",
                    "text": text_value,
                    "is_thought": is_thought,
                    "annotations": serialized_annotations,
                }
            )
    elif content_type == "image":
        raw_data = getattr(item, "data", None)
        if raw_data is None:
            blocks.append(
                {
                    "type": "image_uri",
                    "uri": str(getattr(item, "uri", "") or ""),
                    "is_thought": is_thought,
                }
            )
            return
        try:
            image_bytes = _decode_image_data(raw_data)
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            seen = thought_seen if is_thought else normal_seen
            if image_hash in seen:
                return
            seen.add(image_hash)
            image_tensor, width, height = _image_part_to_tensor(image_bytes)
            (thinking_images if is_thought else normal_images).append(image_tensor)
            artifact_index = len(artifacts)
            artifacts.append(
                ParsedGeminiImageArtifact(
                    mime_type=str(getattr(item, "mime_type", "") or "") or None,
                    raw_bytes=image_bytes,
                    sha256=image_hash,
                    source="interaction_base64",
                    is_thought=is_thought,
                    sequence_index=len(blocks),
                    width=width,
                    height=height,
                )
            )
            blocks.append(
                {
                    "type": "image",
                    "artifact_index": artifact_index,
                    "is_thought": is_thought,
                }
            )
        except Exception as exc:
            diagnostics.append(
                "Malformed Interactions image block was skipped: "
                f"{type(exc).__name__}: {exc}"
            )
            blocks.append(
                {
                    "type": "malformed_image",
                    "is_thought": is_thought,
                    "error": str(exc),
                }
            )
    elif content_type:
        blocks.append(
            {
                "type": "unsupported_content",
                "content_type": content_type,
                "is_thought": is_thought,
            }
        )


def _build_content_pipe(
    *,
    blocks: list[dict[str, Any]],
    artifacts: list[ParsedGeminiImageArtifact],
) -> dict[str, Any]:
    artifact_payloads = []
    for artifact in artifacts:
        artifact_payloads.append(
            {
                "mime_type": artifact.mime_type,
                "raw_bytes": artifact.raw_bytes,
                "sha256": artifact.sha256,
                "source": artifact.source,
                "is_thought": artifact.is_thought,
                "sequence_index": artifact.sequence_index,
                "width": artifact.width,
                "height": artifact.height,
            }
        )
    return {
        "kind": GEMINI_CONTENT_PIPE_KIND,
        "version": GEMINI_CONTENT_PIPE_VERSION,
        "blocks": list(blocks),
        "artifacts": artifact_payloads,
    }


def _serialize_annotations(annotations: Any) -> list[dict[str, Any]]:
    if not annotations:
        return []
    result = []
    for annotation in annotations:
        serialized = _to_jsonable(annotation)
        if isinstance(serialized, dict):
            result.append(serialized)
    return result


def _extract_generate_content_grounding(
    *,
    response: Any,
    candidate: Any,
) -> tuple[dict[str, Any], str]:
    metadata = getattr(candidate, "grounding_metadata", None) if candidate is not None else None
    metadata_dict = _to_jsonable(metadata)
    if not isinstance(metadata_dict, dict):
        metadata_dict = {}
    usage = _to_jsonable(getattr(response, "usage_metadata", None))
    search_suggestions_html = ""
    search_entry_point = metadata_dict.get("search_entry_point")
    if isinstance(search_entry_point, dict):
        rendered = search_entry_point.get("rendered_content")
        if isinstance(rendered, str):
            search_suggestions_html = rendered

    queries = metadata_dict.get("web_search_queries", [])
    if not isinstance(queries, list):
        queries = []
    image_search_steps = metadata_dict.get("image_search_steps", [])
    if not isinstance(image_search_steps, list):
        image_search_steps = []

    sources: list[dict[str, Any]] = []
    chunks = metadata_dict.get("grounding_chunks", [])
    if isinstance(chunks, list):
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            source_payload = None
            source_kind = ""
            for key in ("web", "image", "retrieved_context"):
                candidate_source = chunk.get(key)
                if isinstance(candidate_source, dict):
                    source_payload = candidate_source
                    source_kind = key
                    break
            if source_payload is None:
                continue
            sources.append(
                {
                    "index": index,
                    "kind": source_kind,
                    "title": source_payload.get("title", ""),
                    "url": source_payload.get("uri") or source_payload.get("url") or "",
                    "metadata": source_payload,
                }
            )

    citations: list[dict[str, Any]] = []
    supports = metadata_dict.get("grounding_supports", [])
    if isinstance(supports, list):
        for support in supports:
            if not isinstance(support, dict):
                continue
            indices = support.get("grounding_chunk_indices", [])
            if not isinstance(indices, list):
                indices = []
            segment = support.get("segment", {})
            citations.append(
                {
                    "text": segment.get("text", "") if isinstance(segment, dict) else "",
                    "start_index": segment.get("start_index") if isinstance(segment, dict) else None,
                    "end_index": segment.get("end_index") if isinstance(segment, dict) else None,
                    "source_indices": indices,
                    "sources": [
                        source
                        for source in sources
                        if source.get("index") in indices
                    ],
                    "confidence_scores": support.get("confidence_scores", []),
                }
            )
    return (
        {
            "api_mode": "generate_content",
            "queries": queries,
            "image_search_steps": image_search_steps,
            "sources": sources,
            "citations": citations,
            "grounding": metadata_dict,
            "usage": usage or {},
            "search_suggestions_html": [search_suggestions_html]
            if search_suggestions_html
            else [],
        },
        search_suggestions_html,
    )


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump(mode="json", exclude_none=True))
        except TypeError:
            return _to_jsonable(value.model_dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        return {
            key: _to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }
    return str(value)


def _stringify_response_detail(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    text = str(value).strip()
    return "" if text in {"", "None"} else text


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
