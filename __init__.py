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
                "ComfyUI_Burve_Tools 3.0.0 requires ComfyUI>=0.4.0 with "
                "DynamicCombo and native video input support "
                "with comfy_api.latest available."
            ) from exc
        if exc.name and exc.name.startswith("google.genai"):
            raise RuntimeError(
                "ComfyUI_Burve_Tools 3.0.0 requires google-genai>=2.13.0,<3."
            ) from exc
        raise

    return await _comfy_entrypoint()

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
