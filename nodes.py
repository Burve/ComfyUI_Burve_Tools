import torch
import numpy as np
from PIL import Image
import os
import io
from google import genai
from google.genai import types
import json
import urllib.request
import ssl
import hashlib

# --- Auto-update logic for system instructions ---
INSTRUCTIONS_FILE = os.path.join(os.path.dirname(__file__), "system_instructions.json")
INSTRUCTIONS_URL = "https://raw.githubusercontent.com/Burve/ComfyUI_Burve_Tools/main/system_instructions.json"

def load_instructions():
    if not os.path.exists(INSTRUCTIONS_FILE):
        return {"version": 0, "presets": []}
    try:
        with open(INSTRUCTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[BurveTools] Error loading instructions: {e}")
        return {"version": 0, "presets": []}

def update_instructions():
    print("[BurveTools] Checking for instruction updates...")
    current_data = load_instructions()
    current_version = current_data.get("version", 0)

    try:
        # Create a context that doesn't verify certificates if needed (optional, but good for some envs)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(INSTRUCTIONS_URL, context=ctx, timeout=5) as response:
            if response.status == 200:
                remote_data = json.loads(response.read().decode('utf-8'))
                remote_version = remote_data.get("version", 0)
                
                if remote_version > current_version:
                    print(f"[BurveTools] Updating instructions from v{current_version} to v{remote_version}")
                    with open(INSTRUCTIONS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(remote_data, f, indent=2)
                else:
                    print("[BurveTools] Instructions are up to date.")
    except Exception as e:
        print(f"[BurveTools] Update check failed: {e}")

# Run update on module load
update_instructions()

class BurveGoogleImageGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A futuristic city"}),
                "model": (
                    ["gemini-3-pro-image-preview", "gemini-2.5-flash-image"],
                    {"default": "gemini-2.5-flash-image"},
                ),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "5:4", "4:5", "21:9"],
                    {"default": "1:1"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_google_search": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "system_instructions": ("STRING", {"multiline": True, "default": ""}),
                "reference_images": ("IMAGE_LIST",),
            },
        }

    # normal images, thinking images, thinking text, system messages
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "thinking_image", "thinking_process", "system_messages")
    FUNCTION = "generate_image"
    CATEGORY = "BurveTools"

    # ---- helper to get API key and build a nice error ----
    def _get_api_key_or_error(self):
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            return key, None

        msg = (
            "Gemini API key not found.\n\n"
            "This node reads your key from the GEMINI_API_KEY environment variable.\n\n"
            "Set it like this and then restart ComfyUI:\n\n"
            "Windows (PowerShell):\n"
            "  setx GEMINI_API_KEY \"YOUR_REAL_KEY_HERE\"\n"
            "  # Then close and reopen your terminal / ComfyUI\n\n"
            "macOS / Linux (bash / zsh):\n"
            "  export GEMINI_API_KEY=\"YOUR_REAL_KEY_HERE\"\n"
            "  # If you want it permanent, add that line to ~/.bashrc or ~/.zshrc\n\n"
            "After setting GEMINI_API_KEY and restarting ComfyUI, run this node again."
        )
        return None, msg

    def generate_image(
        self,
        prompt,
        model,
        resolution,
        aspect_ratio,
        seed,
        enable_google_search,
        system_instructions="",
        reference_images=None,
    ):
        # --- API key handling ---
        api_key, key_error = self._get_api_key_or_error()
        if key_error is not None:
            blank = torch.zeros((1, 64, 64, 3))
            # normal_image, thinking_image, thinking_process, system_messages
            return (blank, blank.clone(), "", key_error)

        try:
            client = genai.Client(api_key=api_key)

            contents = [prompt]

            # --- HANDLE REFERENCES AS A LIST ---
            if reference_images is not None:
                for img_tensor in reference_images:
                    if img_tensor is None:
                        continue
                    if not isinstance(img_tensor, torch.Tensor):
                        continue

                    # respect Gemini limit of 14:
                    if len(contents) - 1 >= 14:  # -1 for the text prompt
                        break

                    b = img_tensor.shape[0]
                    for i in range(b):
                        if len(contents) - 1 >= 14:
                            break
                        frame = img_tensor[i]
                        img_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np)
                        contents.append(pil_img)

            # Configure tools
            tools = []
            if enable_google_search:
                tools.append(types.Tool(google_search=types.GoogleSearch()))

            config = None

            if model == "gemini-2.5-flash-image":
                # Per docs: only aspect_ratio, no image_size, no tools, no thinking_config
                image_cfg = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                )
                config = types.GenerateContentConfig(
                    image_config=image_cfg,
                    system_instruction=system_instructions or None,
                    seed=seed % 2147483647 if seed is not None else None,
                )
            elif model == "gemini-3-pro-image-preview":
                # Full config: aspect ratio, resolution, thinking, text+image
                image_cfg = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=resolution,
                )
                thinking_cfg = types.ThinkingConfig(
                    include_thoughts=True,
                    # optional: thinking_budget / thinking_level
                )
                config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=image_cfg,
                    tools=tools if tools else None,
                    system_instruction=system_instructions or None,
                    seed=seed % 2147483647 if seed is not None else None,
                    thinking_config=thinking_cfg,
                )

            # Call API
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            normal_images = []    # non-thinking images
            thinking_images = []  # thinking images (part.thought == True)
            thought_chunks = []
            answer_chunks = []
            system_messages = ""

            # Prefer the canonical structure from docs
            candidates = getattr(response, "candidates", None)
            parts = []

            if candidates:
                # Be defensive: candidates may be empty or content/parts may be None
                first = candidates[0]
                content = getattr(first, "content", None)
                if content is not None:
                    parts = getattr(content, "parts", None) or []
            else:
                # response.parts might exist but be None, so use "or []"
                parts = getattr(response, "parts", None) or []

            # de-dupe using raw bytes so thought + final identical pair collapses to one
            seen_hashes = set()

            for part in parts:
                is_thought = getattr(part, "thought", False)

                # ----- image data -----
                if getattr(part, "inline_data", None):
                    image_bytes = part.inline_data.data

                    img_hash = hashlib.sha256(image_bytes).hexdigest()
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_np = np.array(pil_image).astype(np.float32) / 255.0
                    if image_np.ndim == 2:
                        # grayscale → expand to 3 channels
                        image_np = np.stack([image_np] * 3, axis=-1)

                    image_tensor = torch.from_numpy(image_np).unsqueeze(0)

                    if is_thought:
                        thinking_images.append(image_tensor)
                    else:
                        normal_images.append(image_tensor)

                # ----- text data -----
                txt = getattr(part, "text", None)
                if not txt:
                    continue

                if is_thought:
                    thought_chunks.append(txt)
                else:
                    answer_chunks.append(txt)

            # If there are no normal images but there ARE thinking images,
            # treat the first thinking image as a normal output as well
            if not normal_images and thinking_images:
                normal_images.append(thinking_images[0])

            # Build image batches; ALWAYS return at least 1 image per output
            if normal_images:
                normal_image_batch = torch.cat(normal_images, dim=0)
            else:
                normal_image_batch = torch.zeros((1, 64, 64, 3))
                system_messages = "No non-thinking image generated.\n"

            if thinking_images:
                thinking_image_batch = torch.cat(thinking_images, dim=0)
            else:
                # Return a blank placeholder instead of an empty batch,
                # so SaveImage doesn't crash if connected to this output.
                thinking_image_batch = torch.zeros((1, 64, 64, 3))

            thinking_process = "\n".join(thought_chunks).strip()
            answer_text = "\n".join(answer_chunks).strip()

            # For 3 Pro Image we expect thought summaries when thinking is enabled
            if model == "gemini-3-pro-image-preview":
                if not thinking_process:
                    thinking_process = "No thought summary returned by the model."
            else:
                # 2.5 doesn't support thinking_config, so usually no thought text
                if not thinking_process:
                    thinking_process = ""

            if answer_text:
                if system_messages:
                    system_messages += "\n" + answer_text
                else:
                    system_messages = answer_text

            return (
                normal_image_batch,
                thinking_image_batch,
                thinking_process,
                system_messages,
            )

        except Exception as e:
            blank = torch.zeros((1, 64, 64, 3))
            return (blank, blank.clone(), "", f"Error: {str(e)}")

class BurveImageRefPack:
    @classmethod
    def INPUT_TYPES(s):
        opt = {}
        # 14 optional IMAGE inputs
        for i in range(1, 15):
            opt[f"image{i}"] = ("IMAGE",)

        return {
            "required": {},
            "optional": opt,
        }

    # One output socket, custom type name
    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("images",)
    FUNCTION = "pack"
    CATEGORY = "BurveTools"

    def pack(self, **kwargs):
        """
        kwargs contains image1..image14, each either:
        - None, or
        - tensor [N, H, W, C]
        We return a Python list of tensors. Each item in the list can have its own size.
        """
        result = []

        for key, img in kwargs.items():
            if img is None:
                continue

            if isinstance(img, torch.Tensor):
                # keep batches as-is; Gemini node can iterate frames
                result.append(img)
            else:
                # Just in case someone passes something weird
                raise ValueError(f"{key} is not a tensor: {type(img)}")

        return (result,)

class BurveDebugGeminiKey:
    @classmethod
    def INPUT_TYPES(cls):
        # No inputs – just inspects the environment
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    CATEGORY = "BurveTools/Debug"
    OUTPUT_NODE = True  # can be selected as a workflow output

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force this node to be treated as "changed" every run
        # so Comfy doesn't cache-skip it
        return float("NaN")

    def run(self):
        key = os.getenv("GEMINI_API_KEY", "").strip()

        if not key:
            info = (
                "GEMINI_API_KEY is NOT set or is empty inside ComfyUI.\n\n"
                "Set it in your environment and restart Comfy."
            )
        else:
            info = (
                "GEMINI_API_KEY detected inside ComfyUI:\n"
                f"  length = {len(key)}\n"
                f"  start  = {key[:4]}\n"
                f"  end    = {key[-4:]}\n\n"
                "Full key is NOT shown for safety."
            )

        # Print to terminal
        print("========== [BurveDebugGeminiKey] ==========")
        print(info)
        print("===========================================")

        # Also return it so UI nodes can display it
        return (info,)

class BurveSystemInstructions:
    @classmethod
    def INPUT_TYPES(s):
        data = load_instructions()
        presets = data.get("presets", [])
        # Create a list of titles for the dropdown
        # We'll map titles back to the full instruction in the run method
        # If no presets, provide a placeholder
        if not presets:
            names = ["No instructions found"]
        else:
            names = [p.get("title", "Unknown") for p in presets]
        
        return {
            "required": {
                "instruction_name": (names,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instruction",)
    FUNCTION = "get_instruction"
    CATEGORY = "BurveTools"

    def get_instruction(self, instruction_name):
        data = load_instructions()
        presets = data.get("presets", [])
        
        selected_instruction = ""
        for p in presets:
            if p.get("title") == instruction_name:
                selected_instruction = p.get("system", "")
                break
        
        return (selected_instruction,)

# --- Auto-update logic for prompts ---
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")
PROMPTS_URL = "https://raw.githubusercontent.com/Burve/ComfyUI_Burve_Tools/main/prompts.json"

def load_prompts():
    if not os.path.exists(PROMPTS_FILE):
        return {"version": 0, "prompts": []}
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[BurveTools] Error loading prompts: {e}")
        return {"version": 0, "prompts": []}

def update_prompts():
    print("[BurveTools] Checking for prompt updates...")
    current_data = load_prompts()
    current_version = current_data.get("version", 0)

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(PROMPTS_URL, context=ctx, timeout=5) as response:
            if response.status == 200:
                remote_data = json.loads(response.read().decode('utf-8'))
                remote_version = remote_data.get("version", 0)
                
                if remote_version > current_version:
                    print(f"[BurveTools] Updating prompts from v{current_version} to v{remote_version}")
                    with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(remote_data, f, indent=2)
                else:
                    print("[BurveTools] Prompts are up to date.")
    except Exception as e:
        print(f"[BurveTools] Prompt update check failed: {e}")

# Run update on module load
update_prompts()

class BurveVariableInjector:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {}
        for i in range(1, 15):
            inputs[f"V{i}"] = ("STRING", {"multiline": False, "default": ""})
        
        return {
            "required": {},
            "optional": inputs
        }

    RETURN_TYPES = ("VARIABLE_DICT",)
    RETURN_NAMES = ("variables",)
    FUNCTION = "inject_variables"
    CATEGORY = "BurveTools"

    def inject_variables(self, **kwargs):
        # Filter out empty strings if desired, or keep them. 
        # The user request implies we just pass them all.
        # We'll pass everything that is provided.
        return (kwargs,)

class BurvePromptDatabase:
    @classmethod
    def INPUT_TYPES(cls):
        data = load_prompts()
        prompts = data.get("prompts", [])

        if not prompts:
            names = ["No prompts found"]
        else:
            # generate labels like "[Category] Title"
            labels = []
            for p in prompts:
                title = p.get("title", "Unknown")
                category = p.get("category", "Uncategorized")
                label = f"[{category}] {title}"
                labels.append(label)

            # sort alphabetically (case-insensitive)
            names = sorted(labels, key=lambda x: x.lower())

        return {
            "required": {
                "prompt_name": (names,),
            },
            "optional": {
                # your variables dict input
                "variables": ("VARIABLE_DICT",),
            }
        }

    # now we have a 3rd output for the plain title
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("compiled_prompt", "raw_prompt", "title")
    FUNCTION = "process_prompt"
    CATEGORY = "BurveTools"

    def process_prompt(self, prompt_name, variables=None):
        data = load_prompts()
        prompts = data.get("prompts", [])

        raw_prompt = ""
        raw_title = ""

        for p in prompts:
            title = p.get("title", "Unknown")
            category = p.get("category", "Uncategorized")
            label = f"[{category}] {title}"

            # Compare against the label shown in the dropdown
            if label == prompt_name:
                # store the plain title for output
                raw_title = title
                # In your JSON the field is called "system"
                raw_prompt = p.get("system") or p.get("prompt", "")
                break

        if not raw_prompt:
            return ("", "", "")

        if variables is None:
            variables = {}

        import re
        # Regex to find [[variable_name:default_value]]
        pattern = r"\[\[([a-zA-Z_]\w*):([^\]]*)\]\]"

        def replace_match(match):
            var_name = match.group(1)
            default_val = match.group(2)

            user_value = variables.get(var_name)

            # If user_value is present and not empty, use it. Otherwise default.
            if user_value:
                return str(user_value)
            return default_val

        compiled_prompt = re.sub(pattern, replace_match, raw_prompt)

        # compiled, raw, and the plain title
        return (compiled_prompt, raw_prompt, raw_title)

class BurveBlindGridSplitter:
    """
    Blind grid splitter:
    - Input: IMAGE (B, H, W, C)
    - Params: rows, cols
    - Output: IMAGE batch containing all tiles (B * rows * cols, tile_h, tile_w, C)
    - No OpenCV, no analysis, just straight slicing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 64}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 64}),
                # If image size is not perfectly divisible, you can center-crop
                # the used area instead of from top-left.
                "center_crop": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tiles",)
    FUNCTION = "split"
    CATEGORY = "BurveTools"

    def split(self, image, rows, cols, center_crop=False):
        # image: torch tensor (B, H, W, C)
        b, h, w, c = image.shape

        # Compute tile size from full image size
        tile_h = h // rows
        tile_w = w // cols

        # If tiles would be smaller than 1 px, bail out safely.
        if tile_h < 1 or tile_w < 1:
            print(
                f"BurveBlindGridSplitter: image {h}x{w} too small for "
                f"{rows}x{cols} grid, returning original image."
            )
            return (image,)

        used_h = tile_h * rows
        used_w = tile_w * cols

        # Optionally center-crop the area we actually use
        if center_crop:
            top = max((h - used_h) // 2, 0)
            left = max((w - used_w) // 2, 0)
        else:
            top = 0
            left = 0

        bottom = top + used_h
        right = left + used_w

        cropped = image[:, top:bottom, left:right, :]

        tiles = []
        for bi in range(b):
            for r in range(rows):
                for col in range(cols):
                    y0 = r * tile_h
                    y1 = y0 + tile_h
                    x0 = col * tile_w
                    x1 = x0 + tile_w
                    # Keep batch dim = 1 for each tile so cat works
                    tile = cropped[bi:bi + 1, y0:y1, x0:x1, :]
                    tiles.append(tile)

        if not tiles:
            # Extremely defensive fallback; shouldn't happen.
            return (image,)

        out = torch.cat(tiles, dim=0)
        return (out,)


class BurvePromptSelector14:
    """
    Prompt Selector 14:
    - Stores up to 14 prompt strings
    - Selects one based on index (1–14)
    - Always errors if index is out of bounds (not 1-14)
    - Optional error when selected prompt is empty
    - Outputs: selected_prompt (STRING), selected_index (INT)
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        # 14 multiline prompt inputs
        for i in range(1, 15):
            required[f"prompt_{i}"] = ("STRING", {"multiline": True, "default": ""})

        required["index"] = ("INT", {"default": 1, "min": 1, "max": 14, "step": 1})
        required["error_on_empty"] = ("BOOLEAN", {"default": True})

        return {
            "required": required,
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("selected_prompt", "selected_index")
    FUNCTION = "select"
    CATEGORY = "BurveTools/Text"

    def _cancel_remaining_queue(self):
        """
        Best-effort cancellation of remaining queued runs.
        Wrapped in try/except to avoid crashing if APIs are unavailable.
        """
        # Attempt to interrupt current processing
        try:
            import comfy.model_management
            if hasattr(comfy.model_management, "interrupt_current_processing"):
                comfy.model_management.interrupt_current_processing()
        except Exception:
            pass

        # Attempt to wipe the prompt queue
        try:
            import server
            if hasattr(server, "PromptServer"):
                ps = server.PromptServer.instance
                if ps is not None and hasattr(ps, "prompt_queue"):
                    pq = ps.prompt_queue
                    if hasattr(pq, "wipe"):
                        pq.wipe()
        except Exception:
            pass

    def select(
        self,
        index,
        error_on_empty,
        **kwargs,
    ):
        # Build prompt dict from kwargs (prompt_1 through prompt_14)
        prompts = {}
        for i in range(1, 15):
            key = f"prompt_{i}"
            prompts[i] = kwargs.get(key, "")

        # Always error if index is out of bounds (not between 1 and 14)
        if index < 1 or index > 14:
            self._cancel_remaining_queue()
            raise ValueError(
                f"[BurvePromptSelector14] Index {index} is out of bounds! "
                f"Must be between 1 and 14."
            )

        selected_index = index

        # Get the selected prompt
        selected_prompt = prompts.get(selected_index, "")

        # Check for empty prompt if toggle is enabled
        if error_on_empty and (not selected_prompt or not selected_prompt.strip()):
            # Attempt to cancel remaining queue
            self._cancel_remaining_queue()

            raise ValueError(
                f"[BurvePromptSelector14] Prompt {selected_index} is empty! "
                f"Stopping execution. (error_on_empty=True)"
            )

        return (selected_prompt, selected_index)

