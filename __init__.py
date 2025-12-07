from .nodes import BurveGoogleImageGen, BurveImageRefPack, BurveDebugGeminiKey, BurveSystemInstructions, BurveVariableInjector, BurvePromptDatabase, BurveBlindGridSplitter

NODE_CLASS_MAPPINGS = {
    "BurveGoogleImageGen": BurveGoogleImageGen,
    "BurveImageRefPack": BurveImageRefPack,
    "BurveDebugGeminiKey": BurveDebugGeminiKey,
    "BurveSystemInstructions": BurveSystemInstructions,
    "BurveVariableInjector": BurveVariableInjector,
    "BurvePromptDatabase": BurvePromptDatabase,
    "BurveBlindGridSplitter": BurveBlindGridSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BurveGoogleImageGen": "Burve Google Image Gen",
    "BurveImageRefPack": "Burve Image Reference Pack",
    "BurveDebugGeminiKey": "Burve Debug Gemini Key",
    "BurveSystemInstructions": "Burve System Instructions",
    "BurveVariableInjector": "Burve Variable Injector",
    "BurvePromptDatabase": "Burve Prompt Database",
    "BurveBlindGridSplitter": "Burve Blind Grid Splitter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
