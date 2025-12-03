from .nodes import BurveGoogleImageGen, BurveImageRefPack, BurveDebugGeminiKey

NODE_CLASS_MAPPINGS = {
    "BurveGoogleImageGen": BurveGoogleImageGen,
    "BurveImageRefPack": BurveImageRefPack,
    "BurveDebugGeminiKey": BurveDebugGeminiKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BurveGoogleImageGen": "Burve Google Image Gen",
    "BurveImageRefPack": "Burve Image Ref Pack",
    "BurveDebugGeminiKey": "Burve Debug Gemini Key"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
