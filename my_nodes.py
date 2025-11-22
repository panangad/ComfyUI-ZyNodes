"""
My Custom ComfyUI Nodes - Consolidated
Contains all My* prefixed nodes for better organization
"""

from pathlib import Path
import os
import random
import shutil

from PIL import Image, ImageFilter
from PIL import UnidentifiedImageError
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import comfy.model_management as model_management
import comfy.utils
import cv2
import folder_paths
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF



# ============================================================================
# ============================================================================

class MyIntToString:
    CATEGORY = "Custom/Utils"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"number": ("INT", {"default": 0})}}

    def convert(self, number: int):
        return (str(number),)


class MyCleanStringNode:
    CATEGORY = "Text Processing"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "clean_text"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "default": ""})}}

    def clean_text(self, text):
        cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
        return ("\n".join(cleaned_lines),)


class MyLineSelectorNode:
    CATEGORY = "Custom"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_line"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "line_number": ("INT", {"default": 1, "min": 1})
            }
        }

    def select_line(self, text, line_number):
        lines = [line for line in text.splitlines() if line.strip()]
        if 1 <= line_number <= len(lines):
            return (lines[line_number - 1],)
        return ("",)


class MyAppendToTextFileNode:
    CATEGORY = "Custom"
    RETURN_TYPES = ()
    FUNCTION = "append_to_file"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "example"}),
                "input_float": ("FLOAT", {"default": 0.0}),
            }
        }

    def append_to_file(self, input_string, input_float):
        output_line = f"{input_string},{input_float}\n"
        with open("textout.txt", "a") as f:
            f.write(output_line)
        return ()


# ============================================================================
# ============================================================================

class MyGetLastImage:
    CATEGORY = "Custom"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_last"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",)}}

    def get_last(self, images):
        img = images[-1]
        return (img.unsqueeze(0),)


class MyJoinImageLists:
    CATEGORY = "Custom"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "join"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"list1": ("IMAGE",), "list2": ("IMAGE",)}}

    def join(self, list1, list2):
        s = torch.cat((list1, list2), dim=0)
        return (s,)


class MySaveImage:
    CATEGORY = "image"
    RETURN_TYPES = ()
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    DESCRIPTION = "Save image to ComfyUI output folder (relative paths allowed)."

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "output.png", "multiline": False}),
            }
        }

    def save_image(self, image, filename):
        arr = image[0].cpu().numpy()
        if arr.dtype != np.uint8:
            img_arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            img_arr = arr
        if img_arr.ndim == 3 and img_arr.shape[2] == 1:
            img_arr = img_arr[..., 0]
        out_path = Path(self.output_dir) / Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil = Image.fromarray(img_arr)
        pil.save(str(out_path))
        return ()


class MyZoomImage:
    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "zoom_image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "zoom_percent": ("FLOAT", {"default": 1.0045, "min": 1.0, "max": 5.0, "step": 0.0001}),
            }
        }

    def zoom_image(self, image: torch.Tensor, zoom_percent: float):
        b, h, w, c = image.shape
        result = []
        for i in range(b):
            img = image[i].cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            crop_w = max(1, int(w / zoom_percent))
            crop_h = max(1, int(h / zoom_percent))
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
            cropped = img[y1:y1+crop_h, x1:x1+crop_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            zoomed = torch.from_numpy(zoomed.astype(np.float32) / 255.0)
            result.append(zoomed)
        result = torch.stack(result, dim=0)
        return (result,)


class MyBlend:
    CATEGORY = "image/postprocessing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "imageMask": ("IMAGE",),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["normal", "masked"],),
            }
        }

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, imageMask: torch.Tensor, blend_factor: float, blend_mode: str):
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)
        blended_image = self.blend_mode(image1, image2, imageMask, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def blend_mode(self, img1, img2, imgMask, mode):
        if mode == "normal":
            return img2
        elif mode == "masked":
            return img2 * imgMask + img1 * (1 - imgMask)
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")


class MyTrim:
    CATEGORY = "image/postprocessing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "trim_percentage_top": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "trim_percentage_bottom": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "trim_percentage_left": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "trim_percentage_right": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    def trim(self, image: torch.Tensor, trim_percentage_top: float, trim_percentage_bottom: float, trim_percentage_left: float, trim_percentage_right: float):
        height, width = image.shape[1], image.shape[2]
        trim_top = int((trim_percentage_top / 100.0) * height)
        trim_bottom = int((trim_percentage_bottom / 100.0) * height)
        trim_left = int((trim_percentage_left / 100.0) * width)
        trim_right = int((trim_percentage_right / 100.0) * width)
        return (image[:, trim_top:height-trim_bottom, trim_left:width-trim_right, :],)



class MyZoomAlign:
    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "meta")
    FUNCTION = "zoom_align"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def zoom_align(self, image: torch.Tensor):
        b, h, w, c = image.shape
        zoom = 1.0
        if zoom == 1.0:
            meta = {"zoom": 1.0, "width": w, "height": h}
            return (image, meta)
        crop_w = max(1, int(w / zoom))
        crop_h = max(1, int(h / zoom))
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        result = []
        for i in range(b):
            img = image[i].permute(2, 0, 1).unsqueeze(0)
            cropped = img[:, :, y1:y1+crop_h, x1:x1+crop_w]
            zoomed = comfy.utils.common_upscale(cropped, w, h, upscale_method="bicubic", crop="disabled")
            zoomed = zoomed.squeeze(0).permute(1, 2, 0)
            result.append(zoomed)
        result = torch.stack(result, dim=0)
        meta = {"zoom": zoom, "width": w, "height": h, "crop_w": crop_w, "crop_h": crop_h, "x1": x1, "y1": y1}
        return (result, meta)


class MyZoomReverse:
    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "zoom_reverse"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",), "meta": ("DICT",)}}

    def zoom_reverse(self, image: torch.Tensor, mask: torch.Tensor, meta: dict):
        b, h, w, c = image.shape
        orig_w = meta.get("width", w)
        orig_h = meta.get("height", h)
        crop_w = meta.get("crop_w", w)
        crop_h = meta.get("crop_h", h)
        x1 = meta.get("x1", 0)
        y1 = meta.get("y1", 0)
        restored_images = []
        restored_masks = []
        for i in range(b):
            img = image[i].permute(2, 0, 1).unsqueeze(0)
            resized_img = comfy.utils.common_upscale(img, crop_w, crop_h, upscale_method="bicubic", crop="disabled").squeeze(0).permute(1, 2, 0)
            m = mask[i].unsqueeze(0).unsqueeze(0)
            resized_mask = comfy.utils.common_upscale(m, crop_w, crop_h, upscale_method="nearest-exact", crop="disabled").squeeze(0).squeeze(0)
            canvas_img = torch.zeros((orig_h, orig_w, c), dtype=torch.float32, device=image.device)
            canvas_msk = torch.zeros((orig_h, orig_w), dtype=torch.float32, device=image.device)
            canvas_img[y1:y1+crop_h, x1:x1+crop_w] = resized_img
            canvas_msk[y1:y1+crop_h, x1:x1+crop_w] = resized_mask
            restored_images.append(canvas_img)
            restored_masks.append(canvas_msk)
        restored_images = torch.stack(restored_images, dim=0)
        restored_masks = torch.stack(restored_masks, dim=0)
        return (restored_images, restored_masks)


# ============================================================================
# ============================================================================

class MyDiffMaskNode:
    CATEGORY = "Image Processing"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "make_mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.005}),
                "blur_sigma": ("INT", {"default": 25, "min": 0, "max": 50}),
                "morph_kernel": ("INT", {"default": 20, "min": 1, "max": 51}),
                "min_area": ("INT", {"default": 400, "min": 1}),
                "keep_n": ("INT", {"default": 5, "min": 1, "max": 10}),
            }
        }

    def make_mask(self, image1, image2, threshold, blur_sigma, morph_kernel, min_area, keep_n):
        a = image1[0].cpu().numpy().astype(np.float32)
        b = image2[0].cpu().numpy().astype(np.float32)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[..., :3]
        if b.ndim == 3 and b.shape[2] == 4:
            b = b[..., :3]
        a_f = np.clip(a, 0.0, 1.0).astype(np.float32)
        b_f = np.clip(b, 0.0, 1.0).astype(np.float32)
        if int(blur_sigma) > 0:
            k = max(1, int(blur_sigma) * 2 + 1)
            for ch in range(a_f.shape[2]):
                a_f[..., ch] = cv2.GaussianBlur(a_f[..., ch], (k, k), float(blur_sigma))
                b_f[..., ch] = cv2.GaussianBlur(b_f[..., ch], (k, k), float(blur_sigma))
        diff = np.linalg.norm(a_f - b_f, axis=2)
        mask = (diff > float(threshold)).astype(np.uint8) * 255
        ksize = max(1, int(morph_kernel))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([], dtype=np.int32)
        keep = []
        if areas.size > 0:
            idxs = np.argsort(areas)[::-1]
            for i in idxs:
                if areas[i] >= int(min_area):
                    keep.append(int(i) + 1)
                if len(keep) >= int(keep_n):
                    break
        out = np.zeros_like(mask, dtype=np.uint8)
        for lbl in keep:
            out[labels == lbl] = 1
        out_t = torch.from_numpy(out.astype(np.float32)).unsqueeze(0)
        return (out_t,)


class MyMarkSubjectNode:
    CATEGORY = "image/processing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mark_subject"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "255,0,0"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 10})
            }
        }

    def mark_subject(self, image, mask, color, thickness):
        img = image[0].cpu().numpy()
        msk = mask[0].cpu().numpy()
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        msk_uint8 = (msk * 255).astype(np.uint8)
        contours, _ = cv2.findContours(msk_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        r, g, b = [int(x.strip()) for x in color.split(',')]
        cv2.drawContours(img_uint8, contours, -1, (b, g, r), thickness)
        result = torch.from_numpy(img_uint8.astype(np.float32) / 255.0).unsqueeze(0)
        return (result,)


class MySegmentMask:
    CATEGORY = "Custom"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "getmask"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "boost": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "targetvalues": ("STRING", {"default": "4,5,6,7", "multiline": False}),
                "boostvalues": ("STRING", {"default": "4,5,6,7", "multiline": False}),
            }
        }

    def parse_list(self, s):
        return [int(x.strip()) for x in s.split(',') if x.strip().isdigit()]

    def getmask(self, image, boost, targetvalues, boostvalues):
        processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _, height, width, _ = image.shape
        pil_image = Image.fromarray((image[0].numpy() * 255).astype("uint8"))
        inputs = processor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.cpu()
        tgt_vals = self.parse_list(targetvalues)
        b_vals = self.parse_list(boostvalues)
        target_values = torch.tensor(tgt_vals)
        boost_values = torch.tensor(b_vals)
        for target in boost_values:
            logits[:, target] *= boost
        upsampled_logits = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        binary_mask = torch.where(torch.isin(pred_seg, target_values), 1, 0)
        return (binary_mask.unsqueeze(0),)




class MyReplaceHeadNode:
    CATEGORY = "Image Processing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace_head"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "head_source": ("IMAGE",),
                "x1": ("INT", {"default": 0}),
                "y1": ("INT", {"default": 0}),
                "w1": ("INT", {"default": 100}),
                "h1": ("INT", {"default": 100}),
                "x2": ("INT", {"default": 0}),
                "y2": ("INT", {"default": 0}),
                "w2": ("INT", {"default": 100}),
                "h2": ("INT", {"default": 100}),
            }
        }

    def replace_head(self, base_image, head_source, x1, y1, w1, h1, x2, y2, w2, h2):
        base = base_image.clone()
        head = head_source
        crop = head[:, y2:y2+h2, x2:x2+w2, :]
        crop = crop.permute(0, 3, 1, 2)
        scale = h1 / h2
        new_h = h1
        new_w = int(w2 * scale)
        crop_resized = F.interpolate(crop, size=(new_h, new_w), mode='bilinear', align_corners=False)
        crop_resized = crop_resized.permute(0, 2, 3, 1)
        center_x = x1 + w1 // 2
        center_y = y1 + h1 // 2
        px = max(0, center_x - new_w // 2)
        py = max(0, center_y - new_h // 2)
        _, H, W, _ = base.shape
        px = min(W - new_w, px)
        py = min(H - new_h, py)
        base[0, py:py+new_h, px:px+new_w, :] = crop_resized[0]
        return (base,)


class MyFaceEmbedDistance:
    CATEGORY = "FaceAnalysis"
    RETURN_TYPES = ()
    FUNCTION = "process_folders"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "ref_folder": ("STRING", {"default": "ref"}),
                "image_folder": ("STRING", {"default": "img"}),
                "similarity_metric": (["L2_norm", "cosine", "euclidean"],),
                "output_file": ("STRING", {"default": "d2.txt"}),
            }
        }

    def process_folders(self, analysis_models, ref_folder, image_folder, similarity_metric, output_file):
        if not os.path.isdir(ref_folder) or not os.path.isdir(image_folder):
            raise Exception("Both ref_folder and image_folder must be valid directories.")
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ref_files = [f for f in os.listdir(ref_folder) if os.path.splitext(f)[1].lower() in exts]
        img_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in exts]
        if not ref_files:
            raise Exception("No reference images found in ref_folder.")
        if not img_files:
            raise Exception("No images found in image_folder.")
        with open(output_file, "w") as out_f:
            for ref_name in ref_files:
                print(ref_name)
                ref_path = os.path.join(ref_folder, ref_name)
                try:
                    ref_img = Image.open(ref_path).convert('RGB')
                except Exception:
                    continue
                ref_arr = np.array(ref_img)
                ref_emb = analysis_models.get_embeds(ref_arr)
                if ref_emb is None:
                    continue
                for img_name in img_files:
                    img_path = os.path.join(image_folder, img_name)
                    try:
                        tgt_img = Image.open(img_path).convert('RGB')
                    except Exception:
                        continue
                    tgt_arr = np.array(tgt_img)
                    emb = analysis_models.get_embeds(tgt_arr)
                    if emb is None:
                        continue
                    if similarity_metric == "L2_norm":
                        r = ref_emb / np.linalg.norm(ref_emb)
                        e = emb / np.linalg.norm(emb)
                        dist = float(np.linalg.norm(r - e))
                    elif similarity_metric == "cosine":
                        dist = float(1 - np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb)))
                    else:
                        dist = float(np.linalg.norm(ref_emb - emb))
                    out_f.write(f"{img_name},{dist}\n")
        return ()



# ============================================================================
# ============================================================================

class MyImageCaptioningNode:
    CATEGORY = "Image Processing"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_caption"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "command": ("STRING", {"default": "Give prompt to generate this photo.", "multiline": True}),
            }
        }

    def __init__(self):
        self.device = model_management.get_torch_device()
        model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2"
        ).to(self.device)

    def generate_caption(self, image, command):
        image_pil = Image.fromarray((image[0].cpu().numpy() * 255).astype('uint8'))
        messages = [{"role": "user", "content": [{"type": "image", "image": image_pil}, {"type": "text", "text": command}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=512)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        caption = generated_texts[0]
        caption = caption.split("Assistant:", 1)[-1].strip()
        return (caption,)


class MyQwenImageCaptioningNode:
    CATEGORY = "Image Processing"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_caption"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "command": ("STRING", {"default": "Give prompt to generate this photo.", "multiline": True}),
            },
            "optional": {"image2": ("IMAGE",)}
        }

    def __init__(self):
        self.device = model_management.get_torch_device()
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto").to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.process_vision_info = process_vision_info

    def generate_caption(self, image, command, image2=None):
        image_pil = Image.fromarray((image[0].cpu().numpy() * 255).astype('uint8'))
        content = [{"type": "image", "image": image_pil}, {"type": "text", "text": command}]
        if image2 is not None:
            image_pil2 = Image.fromarray((image2[0].cpu().numpy() * 255).astype('uint8'))
            content = content + [{"type": "image", "image": image_pil2}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = output_text[0]
        caption = caption.split("Assistant:", 1)[-1].strip()
        return (caption,)


class MyQwen3Node:
    CATEGORY = "Text Models"
    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Give 5 prompt to generate a photo"}),
                "model_id": ("STRING", {"default": "Qwen/Qwen3-4B"}),
                "enable_thinking": ("BOOLEAN", {"default": True}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 1000}),
            }
        }

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._loaded = None

    def _ensure(self, model_id):
        if self._loaded == model_id:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self._loaded = model_id

    def generate(self, prompt, model_id, enable_thinking, max_new_tokens, temperature, top_k):
        self._ensure(model_id)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(enable_thinking))
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), temperature=float(temperature), top_k=int(top_k))
        output_ids = gen[0][len(inputs.input_ids[0]):].tolist()
        try:
            idx = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            idx = 0
        thinking_ids = output_ids[:idx]
        content_ids = output_ids[idx:]
        thinking = self.tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
        content = self.tokenizer.decode(content_ids, skip_special_tokens=True).strip()
        return (thinking, content)


class MyFolderFilterNode:
    CATEGORY = "File Processing"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_folder"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "folder1", "multiline": False}),
                "output_folder": ("STRING", {"default": "folder2", "multiline": False}),
                "command": ("STRING", {"default": "Is there a small green rubber band in this photo?", "multiline": True}),
                "expected_output": ("STRING", {"default": "Yes", "multiline": False}),
                "maxl": ("INT", {"default": 100, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {"image2": ("IMAGE",)}
        }

    def __init__(self):
        self.device = model_management.get_torch_device()
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto").to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.process_vision_info = process_vision_info
        self.random = random
        self.shutil = shutil
        self.UnidentifiedImageError = UnidentifiedImageError

    def process_folder(self, input_folder, output_folder, command, expected_output, maxl, image2=None):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        if image2 is not None:
            image_pil2 = Image.fromarray((image2[0].cpu().numpy() * 255).astype('uint8'))
        matched_files = []
        all_files = []
        for root, _, files in os.walk(input_path):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                    all_files.append(Path(root) / fname)
        sampled_files = self.random.sample(all_files, min(maxl, len(all_files)))
        pbar = comfy.utils.ProgressBar(len(sampled_files))
        for file_path in sampled_files:
            print(file_path)
            try:
                pbar.update(1)
                image_pil = Image.open(file_path).convert("RGB")
                w, h = image_pil.size
                scale = (262144 / (w * h)) ** 0.5
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                image_pil = image_pil.resize(new_size, Image.LANCZOS)
                content = [{"type": "image", "image": image_pil}, {"type": "text", "text": command}]
                if image2 is not None:
                    content = content + [{"type": "image", "image": image_pil2}]
                messages = [{"role": "user", "content": content}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = self.process_vision_info(messages)
                inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=1)
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                caption = output_text.split("Assistant:", 1)[-1].strip()
                print(caption)
                if expected_output.lower() in caption.lower():
                    dest = output_path
                    self.shutil.copy(file_path, dest)
                    matched_files.append(str(file_path))
            except (self.UnidentifiedImageError, OSError) as e:
                print(f"Error processing {file_path}: {e}")
                continue
        return (",".join(matched_files),)



# ============================================================================
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "MyIntToString": MyIntToString,
    "MyCleanStringNode": MyCleanStringNode,
    "MyLineSelectorNode": MyLineSelectorNode,
    "MyAppendToTextFileNode": MyAppendToTextFileNode,
    "MyGetLastImage": MyGetLastImage,
    "MyJoinImageLists": MyJoinImageLists,
    "MySaveImage": MySaveImage,
    "ImageMyBlend": MyBlend,
    "MyZoomImage": MyZoomImage,
    "MyZoomAlign": MyZoomAlign,
    "MyZoomReverse": MyZoomReverse,
    "MyTrim": MyTrim,
    "MyDiffMaskNode": MyDiffMaskNode,
    "MyMarkSubjectNode": MyMarkSubjectNode,
    "MySegmentMask": MySegmentMask,
    "MyReplaceHeadNode": MyReplaceHeadNode,
    "MyFaceEmbedDistance": MyFaceEmbedDistance,
    "MyImageCaptioningNode": MyImageCaptioningNode,
    "MyQwenImageCaptioningNode": MyQwenImageCaptioningNode,
    "MyQwen3Node": MyQwen3Node,
    "MyFolderFilterNode": MyFolderFilterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyIntToString": "My Int To String",
    "MyCleanStringNode": "My Clean String",
    "MyLineSelectorNode": "My Line Selector",
    "MyAppendToTextFileNode": "My Append to Text File",
    "MyGetLastImage": "My Get Last Image",
    "MyJoinImageLists": "My Join Image Lists",
    "MySaveImage": "My Save Image",
    "ImageMyBlend": "My Blend",
    "MyZoomImage": "My Zoom Image",
    "MyZoomAlign": "My Zoom Align",
    "MyZoomReverse": "My Zoom Reverse",
    "MyTrim": "My Trim",
    "MyDiffMaskNode": "My Diff Mask",
    "MyMarkSubjectNode": "My Mark Subject",
    "MySegmentMask": "My Segment Mask",
    "MyReplaceHeadNode": "My Replace Head",
    "MyFaceEmbedDistance": "My Face Embeds Distance",
    "MyImageCaptioningNode": "My Image Captioning",
    "MyQwenImageCaptioningNode": "My Qwen Image Captioning",
    "MyQwen3Node": "My Qwen3 Node",
    "MyFolderFilterNode": "My Folder Filter Node",
}

