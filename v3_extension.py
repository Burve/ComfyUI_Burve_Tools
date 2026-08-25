from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from comfy_api.latest import ComfyExtension, IO

try:
    from .gemini_image_service import (
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
        DEFAULT_RETRY_ATTEMPTS,
        GROUNDING_RENDERER_VERSION,
        GEMINI_IMAGE_MODEL_SPECS,
        build_dynamic_combo_options,
        ensure_google_genai_compatibility,
    )
    from .nodes import (
        BurveBlindGridSplitter,
        BurveCharacterPlanner,
        BurveCharacterRaceDetails,
        BurveCropMaskApply,
        BurveCropMaskLoad,
        BurveDebugGeminiKey,
        BurveDebugVertexAuth,
        BurveGoogleImageGen,
        BurveGeminiInputFiles,
        BurveGeminiReferencePack,
        BurveImageInfo,
        BurveImageRefPack,
        BurveListGoogleImageModels,
        BurvePromptDatabase,
        BurvePromptSelector14,
        BurveSaveGeneratedImage,
        BurveSystemInstructions,
        BurveVariableInjector,
        BurveVertexImageGen,
    )
except ImportError:
    from gemini_image_service import (
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
        DEFAULT_RETRY_ATTEMPTS,
        GROUNDING_RENDERER_VERSION,
        GEMINI_IMAGE_MODEL_SPECS,
        build_dynamic_combo_options,
        ensure_google_genai_compatibility,
    )
    from nodes import (
        BurveBlindGridSplitter,
        BurveCharacterPlanner,
        BurveCharacterRaceDetails,
        BurveCropMaskApply,
        BurveCropMaskLoad,
        BurveDebugGeminiKey,
        BurveDebugVertexAuth,
        BurveGoogleImageGen,
        BurveGeminiInputFiles,
        BurveGeminiReferencePack,
        BurveImageInfo,
        BurveImageRefPack,
        BurveListGoogleImageModels,
        BurvePromptDatabase,
        BurvePromptSelector14,
        BurveSaveGeneratedImage,
        BurveSystemInstructions,
        BurveVariableInjector,
        BurveVertexImageGen,
    )


NODE_DEFINITIONS = (
    ("BurveGoogleImageGen", "Burve Google Image Gen", BurveGoogleImageGen),
    ("BurveVertexImageGen", "Burve Google Image Gen (Vertex AI)", BurveVertexImageGen),
    ("BurveImageRefPack", "Burve Image Reference Pack", BurveImageRefPack),
    ("BurveGeminiReferencePack", "Burve Gemini Reference Pack", BurveGeminiReferencePack),
    ("BurveGeminiInputFiles", "Burve Gemini Input Files", BurveGeminiInputFiles),
    ("BurveListGoogleImageModels", "Burve List Google Image Models", BurveListGoogleImageModels),
    ("BurveCropMaskLoad", "Burve Crop + Mask Load", BurveCropMaskLoad),
    ("BurveCropMaskApply", "Burve Crop + Mask Apply", BurveCropMaskApply),
    ("BurveImageInfo", "Burve Image Info", BurveImageInfo),
    ("BurveSaveGeneratedImage", "Burve Save Generated Image", BurveSaveGeneratedImage),
    ("BurveCharacterPlanner", "Burve Character Planner", BurveCharacterPlanner),
    ("BurveCharacterRaceDetails", "Burve Character Race Details", BurveCharacterRaceDetails),
    ("BurveDebugGeminiKey", "Burve Debug Gemini Key", BurveDebugGeminiKey),
    ("BurveDebugVertexAuth", "Burve Debug Vertex Auth", BurveDebugVertexAuth),
    ("BurveSystemInstructions", "Burve System Instructions", BurveSystemInstructions),
    ("BurveVariableInjector", "Burve Variable Injector", BurveVariableInjector),
    ("BurvePromptDatabase", "Burve Prompt Database", BurvePromptDatabase),
    ("BurveBlindGridSplitter", "Burve Blind Grid Splitter", BurveBlindGridSplitter),
    ("BurvePromptSelector14", "Burve Prompt Selector 14", BurvePromptSelector14),
)

BUILTIN_IO_TYPES = {
    "BOOLEAN": IO.Boolean,
    "FLOAT": IO.Float,
    "IMAGE": IO.Image,
    "INT": IO.Int,
    "MASK": IO.Mask,
    "STRING": IO.String,
}


def ensure_dynamic_combo_support() -> None:
    if not hasattr(IO, "DynamicCombo") or not hasattr(IO.DynamicCombo, "Input"):
        raise RuntimeError(
            "ComfyUI_Burve_Tools 3.0.0 requires ComfyUI>=0.4.0 with "
            "DynamicCombo support "
            "with comfy_api.latest.IO.DynamicCombo.Input available."
        )


def _io_type(io_type: str):
    return BUILTIN_IO_TYPES.get(io_type) or IO.Custom(io_type)


def _translate_upload_option(meta: dict[str, Any]) -> Any:
    upload_map = (
        ("image_upload", getattr(getattr(IO, "UploadType", None), "image", None)),
        ("audio_upload", getattr(getattr(IO, "UploadType", None), "audio", None)),
        ("video_upload", getattr(getattr(IO, "UploadType", None), "video", None)),
        ("file_upload", getattr(getattr(IO, "UploadType", None), "model", None)),
    )
    for key, upload_type in upload_map:
        if meta.get(key) and upload_type is not None:
            return upload_type
    return None


def _translate_folder_type(folder_type: Any) -> Any:
    folder_enum = getattr(IO, "FolderType", None)
    if folder_enum is None or folder_type in {None, ""}:
        return None
    if hasattr(folder_enum, str(folder_type)):
        return getattr(folder_enum, str(folder_type))
    return None


def _translate_remote_options(remote: Any) -> Any:
    if remote is None:
        return None
    remote_type = getattr(IO, "RemoteOptions", None)
    if remote_type is None:
        return remote
    if isinstance(remote, dict):
        return remote_type(**remote)
    return remote


def _translate_v1_options(io_type: str, meta: dict[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    extra_dict: dict[str, Any] = {}
    option_map = {
        "advanced": "advanced",
        "default": "default",
        "display_name": "display_name",
        "force_input": "force_input",
        "forceInput": "force_input",
        "label_off": "label_off",
        "label_on": "label_on",
        "lazy": "lazy",
        "max": "max",
        "min": "min",
        "multiline": "multiline",
        "placeholder": "placeholder",
        "raw_link": "raw_link",
        "rawLink": "raw_link",
        "round": "round",
        "socketless": "socketless",
        "step": "step",
        "tooltip": "tooltip",
    }
    upload_flag_keys = {"image_upload", "audio_upload", "video_upload", "file_upload"}

    for key, value in meta.items():
        if key == "control_after_generate":
            translated["control_after_generate"] = value
        elif key in upload_flag_keys or key in {"image_folder", "remote"}:
            continue
        elif key in option_map:
            translated[option_map[key]] = value
        else:
            extra_dict[key] = value

    if io_type == "COMBO":
        upload = _translate_upload_option(meta)
        if upload is not None:
            translated["upload"] = upload

        image_folder = _translate_folder_type(meta.get("image_folder"))
        if image_folder is not None:
            translated["image_folder"] = image_folder

        remote = _translate_remote_options(meta.get("remote"))
        if remote is not None:
            translated["remote"] = remote
    else:
        # ComfyUI's typed upload arguments belong to Combo.Input. Preserve
        # legacy upload metadata for other inputs through extra_dict so their
        # serialized v1 schema remains unchanged without passing unsupported
        # constructor arguments (notably file_upload on String.Input).
        for key in (*upload_flag_keys, "image_folder", "remote"):
            if key in meta:
                extra_dict[key] = meta[key]

    if io_type not in {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}:
        translated.pop("default", None)
        translated.pop("multiline", None)
        translated.pop("placeholder", None)
        translated.pop("min", None)
        translated.pop("max", None)
        translated.pop("step", None)
        translated.pop("round", None)
        translated.pop("label_on", None)
        translated.pop("label_off", None)
        translated.pop("control_after_generate", None)

    if extra_dict:
        translated["extra_dict"] = extra_dict
    return translated


def _build_v1_input(name: str, spec: tuple[Any, ...], optional: bool):
    io_definition = spec[0]
    meta = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}

    if isinstance(io_definition, list):
        kwargs = _translate_v1_options("COMBO", meta)
        return IO.Combo.Input(name, options=list(io_definition), optional=optional, **kwargs)

    comfy_type = _io_type(io_definition)
    kwargs = _translate_v1_options(io_definition, meta)
    return comfy_type.Input(name, optional=optional, **kwargs)


def _build_v1_output(io_type: str, display_name: str, is_output_list: bool):
    comfy_type = _io_type(io_type)
    return comfy_type.Output(id=display_name, display_name=display_name, is_output_list=is_output_list)


def _hidden_token_value(token: Any) -> Any:
    return getattr(token, "value", token)


def _build_v1_hidden(input_types: dict[str, Any]) -> list[Any]:
    hidden_spec = input_types.get("hidden", {})
    if not hidden_spec or not hasattr(IO, "Hidden"):
        return []

    hidden_tokens = []
    available = {
        _hidden_token_value(getattr(IO.Hidden, attr)): getattr(IO.Hidden, attr)
        for attr in (
            "unique_id",
            "prompt",
            "extra_pnginfo",
            "dynprompt",
            "auth_token_comfy_org",
            "api_key_comfy_org",
        )
        if hasattr(IO.Hidden, attr)
    }
    for hidden_type in hidden_spec.values():
        token = available.get(hidden_type)
        if token is not None and token not in hidden_tokens:
            hidden_tokens.append(token)
    return hidden_tokens


@dataclass(frozen=True)
class _NormalizedNodeExecutionResult:
    result: tuple[Any, ...]
    ui: Any = None
    expand: Any = None
    block_execution: Any = None


def _normalize_result(result: Any) -> tuple[Any, ...]:
    if result is None:
        return ()
    if isinstance(result, tuple):
        return result
    return (result,)


def _normalize_node_execution_result(result: Any) -> _NormalizedNodeExecutionResult:
    if isinstance(result, dict):
        return _NormalizedNodeExecutionResult(
            result=_normalize_result(result.get("result")),
            ui=result.get("ui"),
            expand=result.get("expand"),
            block_execution=result.get("block_execution"),
        )
    return _NormalizedNodeExecutionResult(result=_normalize_result(result))


def _create_v1_wrapper(node_id: str, display_name: str, v1_class):
    class WrappedV1Node(IO.ComfyNode):
        V1_CLASS = v1_class

        @classmethod
        def define_schema(cls):
            input_types = cls.V1_CLASS.INPUT_TYPES()
            inputs = []
            for section_name, optional in (("required", False), ("optional", True)):
                for input_name, spec in input_types.get(section_name, {}).items():
                    inputs.append(_build_v1_input(input_name, spec, optional))
            hidden = _build_v1_hidden(input_types)

            return_types = tuple(getattr(cls.V1_CLASS, "RETURN_TYPES", ()))
            return_names = tuple(
                getattr(cls.V1_CLASS, "RETURN_NAMES", return_types)
            )
            output_is_list = tuple(
                getattr(cls.V1_CLASS, "OUTPUT_IS_LIST", [False] * len(return_types))
            )
            outputs = [
                _build_v1_output(io_type, return_names[index], bool(output_is_list[index]))
                for index, io_type in enumerate(return_types)
            ]

            return IO.Schema(
                node_id=node_id,
                display_name=display_name,
                category=getattr(cls.V1_CLASS, "CATEGORY", "BurveTools"),
                inputs=inputs,
                outputs=outputs,
                hidden=hidden,
                is_output_node=bool(getattr(cls.V1_CLASS, "OUTPUT_NODE", False)),
                is_input_list=bool(getattr(cls.V1_CLASS, "INPUT_IS_LIST", False)),
                not_idempotent=bool(getattr(cls.V1_CLASS, "NOT_IDEMPOTENT", False)),
            )

        @classmethod
        def execute(cls, **kwargs):
            instance = cls.V1_CLASS()
            function_name = getattr(cls.V1_CLASS, "FUNCTION")
            result = getattr(instance, function_name)(**kwargs)
            normalized = _normalize_node_execution_result(result)
            return IO.NodeOutput(
                *normalized.result,
                ui=normalized.ui,
                expand=normalized.expand,
                block_execution=normalized.block_execution,
            )

        @classmethod
        def validate_inputs(cls, **kwargs):
            validator = getattr(cls.V1_CLASS, "VALIDATE_INPUTS", None)
            if callable(validator):
                return validator(**kwargs)
            return True

        @classmethod
        def fingerprint_inputs(cls, **kwargs):
            fingerprint = getattr(cls.V1_CLASS, "IS_CHANGED", None)
            if callable(fingerprint):
                return fingerprint(**kwargs)
            return None

        @classmethod
        def check_lazy_status(cls, **kwargs):
            lazy_status = getattr(cls.V1_CLASS, "CHECK_LAZY_STATUS", None)
            if callable(lazy_status):
                return lazy_status(**kwargs)
            return []

    WrappedV1Node.__name__ = f"{v1_class.__name__}V3"
    return WrappedV1Node


def _gemini_outputs():
    return [
        IO.Image.Output(id="image", display_name="image"),
        IO.Image.Output(id="thinking_image", display_name="thinking_image"),
        IO.String.Output(id="thinking_process", display_name="thinking_process"),
        IO.String.Output(id="system_messages", display_name="system_messages"),
        IO.Custom("GENERATED_IMAGE_PIPE").Output(id="generated_image_pipe", display_name="generated_image_pipe"),
        IO.String.Output(id="response_text", display_name="response_text"),
        IO.String.Output(id="grounding_metadata_json", display_name="grounding_metadata_json"),
        IO.Custom("GEMINI_CONTENT_PIPE").Output(id="interleaved_content", display_name="interleaved_content"),
        IO.Custom("GEMINI_INTERACTION_PIPE").Output(id="interaction_pipe", display_name="interaction_pipe"),
    ]


def _video_input():
    video_type = getattr(IO, "Video", None)
    if video_type is not None and hasattr(video_type, "Input"):
        return video_type.Input("video", optional=True)
    return IO.Custom("VIDEO").Input("video", optional=True)


def _gemini_inputs(provider_id):
    inputs = [
        IO.String.Input("prompt", multiline=True, default=""),
        IO.DynamicCombo.Input(
            "model",
            options=build_dynamic_combo_options(IO),
        ),
        IO.Int.Input(
            "seed",
            default=0,
            min=0,
            max=0xFFFFFFFFFFFFFFFF,
            control_after_generate=True,
        ),
        IO.String.Input("system_instructions", multiline=True, default="", optional=True),
        IO.Custom("IMAGE_LIST").Input("reference_images", optional=True),
        IO.Custom("CHARACTER_GEN_PIPE").Input("character_pipe", optional=True),
        IO.String.Input(
            "aspect_ratio_override",
            default="",
            multiline=False,
            optional=True,
            tooltip="Optional override for the selected model's aspect_ratio. Connect Burve Crop + Mask Load here.",
        ),
        IO.Int.Input(
            "request_timeout_seconds",
            default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            min=10,
            max=1800,
            optional=True,
            advanced=True,
            tooltip="Overall timeout budget for the Gemini request, in seconds.",
        ),
        IO.Int.Input(
            "retry_attempts",
            default=DEFAULT_RETRY_ATTEMPTS,
            min=1,
            max=10,
            optional=True,
            advanced=True,
            tooltip="Total attempts for transient Gemini request failures. 1 disables retries.",
        ),
        IO.Combo.Input(
            "api_mode",
            options=["auto", "generate_content", "interactions"],
            default="auto",
            optional=True,
            advanced=True,
        ),
        IO.Combo.Input(
            "response_mode",
            options=["text_and_image", "image_only"],
            default="text_and_image",
            optional=True,
        ),
        IO.Combo.Input(
            "conversation_mode",
            options=["stateless", "stateful"],
            default="stateless",
            optional=True,
            advanced=True,
        ),
        IO.Custom("GEMINI_INTERACTION_PIPE").Input("previous_interaction", optional=True),
        IO.Custom("GEMINI_REFERENCE_PIPE").Input("reference_pack", optional=True),
        IO.Custom("GEMINI_INPUT_FILES_PIPE").Input("input_files", optional=True),
        _video_input(),
        IO.String.Input("video_url", default="", multiline=False, optional=True),
        IO.Float.Input(
            "temperature",
            default=1.0,
            min=0.0,
            max=2.0,
            step=0.05,
            optional=True,
            advanced=True,
        ),
        IO.Float.Input(
            "top_p",
            default=0.95,
            min=0.0,
            max=1.0,
            step=0.01,
            optional=True,
            advanced=True,
        ),
        IO.String.Input(
            "grounding_renderer",
            default="",
            multiline=False,
            optional=True,
            advanced=True,
            tooltip="Managed by the Burve frontend extension for Image Search attribution.",
        ),
    ]
    if provider_id == "vertex":
        inputs.extend(
            [
                IO.Combo.Input(
                    "vertex_location_mode",
                    options=["auto", "configured"],
                    default="auto",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "vertex_output_mime_type",
                    options=[
                        "default",
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/heic",
                        "image/heif",
                    ],
                    default="default",
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input(
                    "output_compression_quality",
                    default=90,
                    min=0,
                    max=100,
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "person_generation",
                    options=["default", "ALLOW_ALL", "ALLOW_ADULT", "ALLOW_NONE"],
                    default="default",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "prominent_people",
                    options=["default", "allow", "block"],
                    default="default",
                    optional=True,
                    advanced=True,
                ),
            ]
        )
    return inputs


def _grounding_ui(result):
    if len(result) < 7:
        return None
    try:
        metadata = json.loads(result[6])
    except (TypeError, ValueError):
        metadata = {}
    if not metadata:
        return {"burve_gemini_grounding": [{"clear": True}]}
    return {
        "burve_gemini_grounding": [
            {
                "renderer_version": GROUNDING_RENDERER_VERSION,
                "metadata": metadata,
            }
        ]
    }


class BurveGoogleImageGenV3(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BurveGoogleImageGen",
            display_name="Burve Google Image Gen",
            category=BurveGoogleImageGen.CATEGORY,
            inputs=_gemini_inputs("aistudio"),
            outputs=_gemini_outputs(),
        )

    @classmethod
    def execute(
        cls,
        prompt,
        model,
        seed,
        system_instructions="",
        reference_images=None,
        character_pipe=None,
        aspect_ratio_override="",
        request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        retry_attempts=DEFAULT_RETRY_ATTEMPTS,
        api_mode="auto",
        response_mode="text_and_image",
        conversation_mode="stateless",
        previous_interaction=None,
        reference_pack=None,
        input_files=None,
        video=None,
        video_url="",
        temperature=1.0,
        top_p=0.95,
        grounding_renderer="",
    ):
        node = BurveGoogleImageGen()
        result = node.generate_image(
            prompt=prompt,
            model=model,
            seed=seed,
            system_instructions=system_instructions,
            reference_images=reference_images,
            character_pipe=character_pipe,
            aspect_ratio_override=aspect_ratio_override,
            request_timeout_seconds=request_timeout_seconds,
            retry_attempts=retry_attempts,
            api_mode=api_mode,
            response_mode=response_mode,
            conversation_mode=conversation_mode,
            previous_interaction=previous_interaction,
            reference_pack=reference_pack,
            input_files=input_files,
            video=video,
            video_url=video_url,
            temperature=temperature,
            top_p=top_p,
            grounding_renderer=grounding_renderer,
        )
        return IO.NodeOutput(
            *result,
            ui=_grounding_ui(result),
        )


class BurveVertexImageGenV3(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BurveVertexImageGen",
            display_name="Burve Google Image Gen (Vertex AI)",
            category=BurveVertexImageGen.CATEGORY,
            inputs=_gemini_inputs("vertex"),
            outputs=_gemini_outputs(),
        )

    @classmethod
    def execute(
        cls,
        prompt,
        model,
        seed,
        system_instructions="",
        reference_images=None,
        character_pipe=None,
        aspect_ratio_override="",
        request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        retry_attempts=DEFAULT_RETRY_ATTEMPTS,
        api_mode="auto",
        response_mode="text_and_image",
        conversation_mode="stateless",
        previous_interaction=None,
        reference_pack=None,
        input_files=None,
        video=None,
        video_url="",
        temperature=1.0,
        top_p=0.95,
        grounding_renderer="",
        vertex_location_mode="auto",
        vertex_output_mime_type="default",
        output_compression_quality=90,
        person_generation="default",
        prominent_people="default",
    ):
        node = BurveVertexImageGen()
        result = node.generate_image(
            prompt=prompt,
            model=model,
            seed=seed,
            system_instructions=system_instructions,
            reference_images=reference_images,
            character_pipe=character_pipe,
            aspect_ratio_override=aspect_ratio_override,
            request_timeout_seconds=request_timeout_seconds,
            retry_attempts=retry_attempts,
            api_mode=api_mode,
            response_mode=response_mode,
            conversation_mode=conversation_mode,
            previous_interaction=previous_interaction,
            reference_pack=reference_pack,
            input_files=input_files,
            video=video,
            video_url=video_url,
            temperature=temperature,
            top_p=top_p,
            grounding_renderer=grounding_renderer,
            vertex_location_mode=vertex_location_mode,
            vertex_output_mime_type=vertex_output_mime_type,
            output_compression_quality=output_compression_quality,
            person_generation=person_generation,
            prominent_people=prominent_people,
        )
        return IO.NodeOutput(
            *result,
            ui=_grounding_ui(result),
        )


NON_GEMINI_V3_NODES = [
    _create_v1_wrapper(node_id, display_name, cls)
    for node_id, display_name, cls in NODE_DEFINITIONS
    if node_id not in {"BurveGoogleImageGen", "BurveVertexImageGen"}
]


class BurveExtension(ComfyExtension):
    async def on_load(self) -> None:
        ensure_dynamic_combo_support()
        ensure_google_genai_compatibility()

    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        ensure_dynamic_combo_support()
        ensure_google_genai_compatibility()
        return [
            BurveGoogleImageGenV3,
            BurveVertexImageGenV3,
            *NON_GEMINI_V3_NODES,
        ]


async def comfy_entrypoint() -> BurveExtension:
    ensure_dynamic_combo_support()
    ensure_google_genai_compatibility()
    return BurveExtension()
