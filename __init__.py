WEB_DIRECTORY = "./web/js"


async def comfy_entrypoint():
    try:
        try:
            from .v3_extension import comfy_entrypoint as _comfy_entrypoint
        except ImportError:
            from v3_extension import comfy_entrypoint as _comfy_entrypoint
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("comfy_api"):
            raise RuntimeError(
                "ComfyUI_Burve_Tools 2.2.0 requires a DynamicCombo-capable ComfyUI build "
                "with comfy_api.latest available."
            ) from exc
        if exc.name and exc.name.startswith("google.genai"):
            raise RuntimeError(
                "ComfyUI_Burve_Tools 2.2.0 requires google-genai>=1.68.0,<2."
            ) from exc
        raise

    return await _comfy_entrypoint()

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
