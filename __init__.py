from .nodes import BurveGoogleImageGen, BurveVertexImageGen, BurveImageRefPack, BurveCharacterPlanner, BurveCharacterRaceDetails, BurveDebugGeminiKey, BurveDebugVertexAuth, BurveSystemInstructions, BurveVariableInjector, BurvePromptDatabase, BurveBlindGridSplitter, BurvePromptSelector14

NODE_CLASS_MAPPINGS = {
    "BurveGoogleImageGen": BurveGoogleImageGen,
    "BurveVertexImageGen": BurveVertexImageGen,
    "BurveImageRefPack": BurveImageRefPack,
    "BurveCharacterPlanner": BurveCharacterPlanner,
    "BurveCharacterRaceDetails": BurveCharacterRaceDetails,
    "BurveDebugGeminiKey": BurveDebugGeminiKey,
    "BurveDebugVertexAuth": BurveDebugVertexAuth,
    "BurveSystemInstructions": BurveSystemInstructions,
    "BurveVariableInjector": BurveVariableInjector,
    "BurvePromptDatabase": BurvePromptDatabase,
    "BurveBlindGridSplitter": BurveBlindGridSplitter,
    "BurvePromptSelector14": BurvePromptSelector14,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BurveGoogleImageGen": "Burve Google Image Gen",
    "BurveVertexImageGen": "Burve Google Image Gen (Vertex AI)",
    "BurveImageRefPack": "Burve Image Reference Pack",
    "BurveCharacterPlanner": "Burve Character Planner",
    "BurveCharacterRaceDetails": "Burve Character Race Details",
    "BurveDebugGeminiKey": "Burve Debug Gemini Key",
    "BurveDebugVertexAuth": "Burve Debug Vertex Auth",
    "BurveSystemInstructions": "Burve System Instructions",
    "BurveVariableInjector": "Burve Variable Injector",
    "BurvePromptDatabase": "Burve Prompt Database",
    "BurveBlindGridSplitter": "Burve Blind Grid Splitter",
    "BurvePromptSelector14": "Burve Prompt Selector 14",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
