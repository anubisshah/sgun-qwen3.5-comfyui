import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
import os
import folder_paths
import numpy as np
from PIL import Image

class Qwen35Loader:
    @classmethod
    def INPUT_TYPES(s):
        # 扫描 ComfyUI/models/LLM 目录下的本地模型
        model_dir = os.path.join(folder_paths.models_dir, "LLM")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            
        local_models = []
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                if os.path.isdir(os.path.join(model_dir, item)):
                    local_models.append(item)
        
        # 仅保留官方已发布的 Base 项
        default_models = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B"]
        model_list = local_models + default_models

        return {
            "required": {
                "model_name": (model_list, {"default": model_list[0] if model_list else "Qwen/Qwen3.5-9B"}),
                "download_if_missing": ("BOOLEAN", {"default": False}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "local_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL", "QWEN_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "Qwen3.5"

    def load_model(self, model_name, download_if_missing, device, precision, local_path=""):
        base_model_dir = os.path.join(folder_paths.models_dir, "LLM")
        
        if local_path and os.path.exists(local_path):
            model_id = local_path
        else:
            potential_local_path = os.path.join(base_model_dir, model_name)
            if os.path.exists(potential_local_path) and os.path.isdir(potential_local_path):
                model_id = potential_local_path
            else:
                repo_name = model_name.split("/")[-1]
                target_path = os.path.join(base_model_dir, repo_name)
                
                if os.path.exists(target_path):
                     model_id = target_path
                elif download_if_missing:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=model_name, local_dir=target_path)
                    model_id = target_path
                else:
                    model_id = model_name

        dtype = torch.float16
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp32":
            dtype = torch.float32

        device_map = device if device != "auto" else "auto"
        
        try:
            try:
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            except Exception:
                processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            has_vision = hasattr(config, "vision_config") and config.vision_config is not None
            if has_vision:
                import transformers
                model_class = None
                if hasattr(config, "architectures") and config.architectures:
                    arch_name = config.architectures[0]
                    model_class = getattr(transformers, arch_name, None)
                if model_class is None:
                    model_class = AutoModelForCausalLM
                model = model_class.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
                model.is_vision_model = model_class is not AutoModelForCausalLM
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
                model.is_vision_model = False
        except Exception as e:
            raise ValueError(f"Error loading model {model_id}: {e}")

        return (model, processor)

class Qwen35Generator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN_MODEL",),
                "processor": ("QWEN_PROCESSOR",),
                "prompt": ("STRING", {"multiline": True, "default": "这张图的内容是："}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 100}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3.5"

    def generate(self, model, processor, prompt, system_prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, seed, image=None):
        torch.manual_seed(seed)
        
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        user_content = []
        if image is not None:
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            user_content.append({"type": "image", "image": img})
        
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        if image is not None and hasattr(processor, "image_processor") and getattr(model, "is_vision_model", False):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        else:
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") and processor.tokenizer is not None else processor
            text = prompt
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            if "pixel_values" in inputs: del inputs["pixel_values"]
            if "image_grid_thw" in inputs: del inputs["image_grid_thw"]

        generate_kwargs = inputs if (image is not None and getattr(model, "is_vision_model", False)) else {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

        with torch.no_grad():
            generated_ids = model.generate(
                **generate_kwargs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 0 else False
            )
        
        generated_ids_trimmed = [out[len(ins):] for ins, out in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0] if hasattr(processor, "batch_decode") else processor.decode(generated_ids_trimmed[0], skip_special_tokens=True)
            
        # 仅清理碎片，不释放模型
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return (response,)

NODE_CLASS_MAPPINGS = {"Qwen35Loader": Qwen35Loader, "Qwen35Generator": Qwen35Generator}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen35Loader": "Qwen 3.5 Model Loader", "Qwen35Generator": "Qwen 3.5 Generator"}