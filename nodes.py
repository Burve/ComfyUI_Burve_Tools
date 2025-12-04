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
                "model": (["gemini-3-pro-image-preview", "gemini-2.5-flash-image"], {"default": "gemini-2.5-flash-image"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "5:4", "4:5", "21:9"], {"default": "1:1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_google_search": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "system_instructions": ("STRING", {"multiline": True, "default": ""}),
                "reference_images": ("IMAGE_LIST",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "thinking_process", "system_messages")
    FUNCTION = "generate_image"
    CATEGORY = "BurveTools"

        # ---- helper to get API key and build a nice error ----
    def _get_api_key_or_error(self):
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            return key, None

        # This text will be shown in system_messages if the key is missing
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

    def generate_image(self, prompt, model, resolution, aspect_ratio, seed, enable_google_search, system_instructions="", reference_images=None):
        # --- API key handling ---
        api_key, key_error = self._get_api_key_or_error()
        if key_error is not None:
            # Return a tiny blank image + the instructions as system_messages
            blank = torch.zeros((1, 64, 64, 3))
            return (blank, "", key_error)

        try:
            client = genai.Client(api_key=api_key)
            
            contents = [prompt]
            
            # --- HANDLE REFERENCES AS A LIST ---
            if reference_images is not None:
                # reference_images is a Python list of tensors
                # each tensor: [N, H, W, C] (N might be 1)
                for img_tensor in reference_images:
                    if img_tensor is None:
                        continue
                    if not isinstance(img_tensor, torch.Tensor):
                        continue

                    # respect Gemini limit of 14:
                    if len(contents) - 1 >= 14:  # -1 for the text prompt
                        break

                    # iterate frames in batch
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

            # Configure generation
            config = types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=resolution,
                ),
                tools=tools if tools else None,
                system_instruction=system_instructions if system_instructions else None,
                seed=seed % 2147483647 if seed is not None else None,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    # optional: thinking_budget=1024 or thinking_level="low"/"high" on 3.x models
                ),
            )

            # Call API
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            output_image = None
            thinking_process = ""
            system_messages = ""

            # Prefer the canonical structure from docs
            candidates = getattr(response, "candidates", None)
            if candidates:
                parts = candidates[0].content.parts
            else:
                # fallback for convenience alias
                parts = getattr(response, "parts", [])

            thought_chunks = []
            answer_chunks = []

            for part in parts:
                # image data
                if getattr(part, "inline_data", None):
                    image_bytes = part.inline_data.data
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    image_np = np.array(pil_image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                    if output_image is None:
                        output_image = image_tensor
                    else:
                        output_image = torch.cat((output_image, image_tensor), dim=0)

                # text data
                txt = getattr(part, "text", None)
                if not txt:
                    continue

                if getattr(part, "thought", False):
                    # this is the “thinking” summary
                    thought_chunks.append(txt)
                else:
                    # this is normal answer text (you can decide where to send it)
                    answer_chunks.append(txt)

            thinking_process = "\n".join(thought_chunks).strip()
            answer_text = "\n".join(answer_chunks).strip()

            if output_image is None:
                output_image = torch.zeros((1, 64, 64, 3))
                system_messages = "No image generated.\n"
            else:
                system_messages = answer_text  # or keep this for errors only, up to you

            # If absolutely nothing texty came back, at least say that
            if not thinking_process:
                thinking_process = "No thought summary returned by the model."

            return (output_image, thinking_process, system_messages)

        except Exception as e: return (torch.zeros((1, 64, 64, 3)), "", f"Error: {str(e)}")

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
                "variables": ("VARIABLE_DICT",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("compiled_prompt", "raw_prompt")
    FUNCTION = "process_prompt"
    CATEGORY = "BurveTools"

    def process_prompt(self, prompt_name, variables=None):
        data = load_prompts()
        prompts = data.get("prompts", [])

        raw_prompt = ""

        for p in prompts:
            title = p.get("title", "Unknown")
            category = p.get("category", "Uncategorized")
            label = f"[{category}] {title}"

            # Now we compare against the *label* shown in the dropdown
            if label == prompt_name:
                # In your JSON the field is called "system"
                raw_prompt = p.get("system") or p.get("prompt", "")
                break

        if not raw_prompt:
            return ("", "")

        if variables is None:
            variables = {}

        # Regex to find [[variable_name:default_value]]
        # Group 1: variable_name
        # Group 2: default_value
        import re
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

        return (compiled_prompt, raw_prompt)
