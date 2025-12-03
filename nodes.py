import torch
import numpy as np
from PIL import Image
import os
import io
from google import genai
from google.genai import types

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
                seed=seed % 2147483647 if seed is not None else None
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

            # Parse response
            if response.parts:
                for part in response.parts:
                    if part.text:
                        # Check if it's thinking process or regular text
                        # The API might return thinking process in a specific way, but usually it's just text parts
                        # For now, we append all text to thinking_process/system_messages
                        # If the model refuses, it often comes as text.
                        # We'll put all text in thinking_process for now, or check for specific flags if documented.
                        # The documentation mentions "Thinking mode" generates interim thought images but doesn't explicitly say text is separated differently in the parts list for the final response, 
                        # but often reasoning models output text first.
                        thinking_process += part.text + "\n"
                    
                    if part.inline_data:
                        # Convert bytes to PIL Image
                        image_bytes = part.inline_data.data
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # Convert PIL to Tensor
                        # PIL image is RGB, convert to numpy float [0, 1]
                        image_np = np.array(pil_image).astype(np.float32) / 255.0
                        # Add batch dimension [1, H, W, C]
                        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                        
                        if output_image is None:
                            output_image = image_tensor
                        else:
                            # If multiple images, concatenate? Usually just one.
                            output_image = torch.cat((output_image, image_tensor), dim=0)
            
            if output_image is None:
                # If no image generated, return a blank one and error message
                output_image = torch.zeros((1, 64, 64, 3))
                system_messages += "No image generated. Check thinking process for details.\n"

            return (output_image, thinking_process, system_messages)

        except Exception as e:
            return (torch.zeros((1, 64, 64, 3)), "", f"Error: {str(e)}")

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