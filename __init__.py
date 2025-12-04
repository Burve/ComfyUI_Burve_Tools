from .nodes import BurveGoogleImageGen, BurveImageRefPack, BurveDebugGeminiKey, BurveSystemInstructions, BurveVariableInjector, BurvePromptDatabase

NODE_CLASS_MAPPINGS = {
    "BurveGoogleImageGen": BurveGoogleImageGen,
    "BurveImageRefPack": BurveImageRefPack,
    "BurveDebugGeminiKey": BurveDebugGeminiKey,
    "BurveSystemInstructions": BurveSystemInstructions,
    "BurveVariableInjector": BurveVariableInjector,
    "BurvePromptDatabase": BurvePromptDatabase,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BurveGoogleImageGen": "Burve Google Image Gen",
    "BurveImageRefPack": "Burve Image Reference Pack",
    "BurveDebugGeminiKey": "Burve Debug Gemini Key",
    "BurveSystemInstructions": "Burve System Instructions",
    "BurveVariableInjector": "Burve Variable Injector",
    "BurvePromptDatabase": "Burve Prompt Database",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
