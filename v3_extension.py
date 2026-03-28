from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comfy_api.latest import ComfyExtension, IO

try:
    from .gemini_image_service import (
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
        DEFAULT_RETRY_ATTEMPTS,
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
        BurveImageInfo,
        BurveImageRefPack,
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
        BurveImageInfo,
        BurveImageRefPack,
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
            "ComfyUI_Burve_Tools 2.1.0 requires a DynamicCombo-capable ComfyUI build "
            "with comfy_api.latest.IO.DynamicCombo.Input available."
        )


def _io_type(io_type: str):
    return BUILTIN_IO_TYPES.get(io_type) or IO.Custom(io_type)


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

    for key, value in meta.items():
        if key == "control_after_generate":
            translated["control_after_generate"] = value
        elif key in option_map:
            translated[option_map[key]] = value
        else:
            extra_dict[key] = value

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
    ]


def _gemini_inputs():
    return [
        IO.String.Input("prompt", multiline=True, default=""),
        IO.DynamicCombo.Input("model", options=build_dynamic_combo_options(IO)),
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
    ]


class BurveGoogleImageGenV3(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BurveGoogleImageGen",
            display_name="Burve Google Image Gen",
            category=BurveGoogleImageGen.CATEGORY,
            inputs=_gemini_inputs(),
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
    ):
        node = BurveGoogleImageGen()
        return IO.NodeOutput(
            *node.generate_image(
                prompt=prompt,
                model=model,
                seed=seed,
                system_instructions=system_instructions,
                reference_images=reference_images,
                character_pipe=character_pipe,
                aspect_ratio_override=aspect_ratio_override,
                request_timeout_seconds=request_timeout_seconds,
                retry_attempts=retry_attempts,
            )
        )


class BurveVertexImageGenV3(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BurveVertexImageGen",
            display_name="Burve Google Image Gen (Vertex AI)",
            category=BurveVertexImageGen.CATEGORY,
            inputs=_gemini_inputs(),
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
    ):
        node = BurveVertexImageGen()
        return IO.NodeOutput(
            *node.generate_image(
                prompt=prompt,
                model=model,
                seed=seed,
                system_instructions=system_instructions,
                reference_images=reference_images,
                character_pipe=character_pipe,
                aspect_ratio_override=aspect_ratio_override,
                request_timeout_seconds=request_timeout_seconds,
                retry_attempts=retry_attempts,
            )
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
