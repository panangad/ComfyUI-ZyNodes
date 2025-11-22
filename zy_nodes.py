import math
import random

from PIL import Image, ImageFilter
from scipy.stats import linregress
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
import comfy.utils
import cv2
import numpy as np
import numpy as np, cv2, torch
import numpy as np, torch, scipy.ndimage
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def fast_blur(m, kernel_size, scale=0.25):
    h, w = m.shape[2:]
    m_small = F.interpolate(m, scale_factor=scale, mode="bilinear", align_corners=False)
    m_small = TF.gaussian_blur(m_small, kernel_size)
    # m_small = F.avg_pool2d(m_small, kernel_size, stride=1, padding=kernel_size//2)
    return F.interpolate(m_small, (h, w), mode="bilinear", align_corners=False)

class CropByMaskBBox:
    TARGET_SIZES = [(1024,1024),(880,1184),(832,1248),(768,1376),(1184,880),(1248,832),(1376,768)]
    # TARGET_SIZES = [(1376, 1376),(1216, 1536),(1024, 1664),(896, 1728),(1536, 1216),(1664, 1024),(1728, 896)]

    @classmethod
    def INPUT_TYPES(cls): return {"required":{"image":("IMAGE",),"mask":("MASK",), "padding": ("INT", {"default": 10, "min": 1, "max": 201, "step": 1}),}}
    RETURN_TYPES = ("IMAGE","TENSOR",)
    RETURN_NAMES = ("resized_cropped","stitch_info")
    FUNCTION = "crop"
    CATEGORY = "image/processing"

    def mask_bbox(self, mask, padding=10):
        m = mask[0]
        rows = torch.any(m > 0, dim=1)
        cols = torch.any(m > 0, dim=0)
        if not rows.any() or not cols.any():
            return 0, 0, 1, 1
        y0, y1 = torch.where(rows)[0][[0, -1]]
        x0, x1 = torch.where(cols)[0][[0, -1]]
        h, w = m.shape
        x0 = max(0, x0 - int(padding/4))
        y0 = max(0, y0 - int(padding/4) )
        x1 = min(w - 1, x1 + int(padding/4))
        y1 = min(h - 1, y1 + padding)
        return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


    def fit_aspect_rect(self, Mw, Mh, cx, cy, cw, ch, rw, rh):
        def expand(w,h,rw,rh):
            tw = max(w, -(-h*rw//rh))
            th = max(h, -(-w*rh//rw))
            if tw*rh >= th*rw:
                th = tw*rh//rw
            else:
                tw = th*rw//rh
            return tw, th
        rw_, rh_ = expand(cw, ch, rw, rh)
        if rw_ > Mw or rh_ > Mh:
            return None
        ccx, ccy = cx + cw//2, cy + ch//2
        x = max(0, min(Mw - rw_, ccx - rw_//2))
        y = max(0, min(Mh - rh_, ccy - rh_//2))
        return x, y, rw_, rh_

    def max_box_for_aspect(self, Mw, Mh, rw, rh):
        if Mw * rh <= Mh * rw:
            bw = Mw
            bh = (Mw * rh) // rw
        else:
            bh = Mh
            bw = (Mh * rw) // rh
        x = max(0, (Mw - bw) // 2)
        y = Mh - bh
        return int(x), int(y), int(bw), int(bh)


    def crop(self, image, mask, padding):
        img = image[0]
        Mh, Mw = img.shape[0], img.shape[1]
        padding = int((padding / 100) * Mw)
        mx, my, mw, mh = self.mask_bbox(mask, padding)
        chosen = None
        chosen_target = None
        for rw, rh in self.TARGET_SIZES:
            r = self.fit_aspect_rect(Mw, Mh, mx, my, mw, mh, rw, rh)
            if r is not None:
                chosen = r
                chosen_target = (rw, rh)
                break
        if chosen is None:
            # chosen = self.max_box_for_aspect(Mw, Mh, *self.TARGET_SIZES[1])
            mask_ar = (mw / mh)
            best_target = min(self.TARGET_SIZES, key=lambda t: abs((t[0] / t[1]) - mask_ar))
            chosen = self.max_box_for_aspect(Mw, Mh, *best_target)
            chosen_target = best_target
        bx, by, bw, bh = chosen
        tw, th = chosen_target
        crop = img[by:by+bh, bx:bx+bw].permute(2,0,1).unsqueeze(0).contiguous()
        resized = F.interpolate(crop, size=(th, tw), mode="bilinear", align_corners=False)
        out = resized.permute(0,2,3,1).contiguous()
        stitch_info = torch.tensor([int(bx), int(by), int(bw), int(bh), int(tw), int(th)])
        return out, stitch_info



class StitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch_info": ("TENSOR",),
                "original": ("IMAGE",),
                "replacement": ("IMAGE",),
                "mask": ("MASK",),
                "radius": ("INT", {"default": 5, "min": 1, "max": 201, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "image/processing"


    def stitch(self, stitch_info, original, replacement, mask, radius=5):
        x1, y1, crop_w, crop_h = [int(v.item()) for v in stitch_info[:4]]
        orig = original[0].unsqueeze(0)
        rep = replacement[0].unsqueeze(0)
        m = mask[0].unsqueeze(0)
        if m.ndim == 3:
            m = m.unsqueeze(-1)

        kernel_size = int((radius / 1000.0) * crop_w)
        if kernel_size < 3: kernel_size = 3
        if kernel_size % 2 == 0: kernel_size += 1

        rep = F.interpolate(rep.permute(0,3,1,2), (crop_h, crop_w), mode="bilinear", align_corners=False).permute(0,2,3,1)
        m = F.interpolate(m.permute(0,3,1,2).float(), (crop_h, crop_w), mode="bilinear", align_corners=False)
        # m = F.avg_pool2d(m, kernel_size, stride=1, padding=kernel_size//2)
        m = fast_blur(m,kernel_size,0.2)

        m = m.permute(0,2,3,1).repeat(1,1,1,rep.shape[-1])

        crop = orig[:, y1:y1+crop_h, x1:x1+crop_w, :]
        blended = crop * (1 - m) + rep * m
        out = orig.clone()
        out[:, y1:y1+crop_h, x1:x1+crop_w, :] = blended
        return (out,)


class TrimImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "stitch_info": ("TENSOR",),
                "trim_percent": ("FLOAT", {
                    "default": 0.05,  # 5%
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.001
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", )
    FUNCTION = "trim_image"
    CATEGORY = "image/transform"

    def trim_image(self, image: torch.Tensor, stitch_info, trim_percent: float):
        # y0, x0, crop_h, crop_w, target_h, target_w = [int(x) for x in stitch_info.tolist()]
        x0, y0, crop_w, crop_h, target_w, target_h = [int(x) for x in stitch_info.tolist()]
        print(target_h)
        _, h, w, c = image.shape


        trim_top = y0 < 3
        trim_bottom = abs(y0 + crop_h - h) < 3
        trim_left = x0 < 3
        trim_right = abs(x0 + crop_w - w) < 3

        trim_px_top = int(h * trim_percent) if trim_top else 0
        trim_px_bottom = int(h * trim_percent) if trim_bottom else 0
        trim_px_left = int(w * trim_percent) if trim_left else 0
        trim_px_right = int(w * trim_percent) if trim_right else 0

        y1 = max(0, trim_px_top)
        y2 = min(h, h - trim_px_bottom)
        x1 = max(0, trim_px_left)
        x2 = min(w, w - trim_px_right)

        trimmed = image[0, y1:y2, x1:x2, :]

        trimmed = trimmed.unsqueeze(0)

        intvl = (int(not trim_bottom) << 0) | (int(not trim_left) << 1) | (int(not trim_right) << 2) | (int(not trim_top) << 3)

        return (trimmed, intvl,)


class MaskModifier:
    CATEGORY = "image/transform"
    FUNCTION = "modify_mask"
    RETURN_TYPES = ("MASK",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "flags": ("INT", {"default": 1}),
                "percent_to_fill": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    def modify_mask(self, mask, flags, percent_to_fill):
        trim_bottom = bool(flags & (1 << 0))
        trim_left   = bool(flags & (1 << 1))
        trim_right  = bool(flags & (1 << 2))
        trim_top    = bool(flags & (1 << 3))

        if flags == 99:
            return (mask,)

        C, height, width = mask.shape
        p = max(0.0, min(100.0, float(percent_to_fill)))
        cutoff_row = int(height * (1.0 - (p / 100.0)))
        cutoff_col = int(width  * (1.0 - (p / 100.0)))

        m = mask.clone()

        if trim_bottom and 0 < cutoff_row < height:
            m[:, cutoff_row:, :] = 0.0
        top_cut = height - cutoff_row
        if trim_top and 0 < top_cut < height:
            m[:, :top_cut, :] = 0.0
        if trim_right and 0 < cutoff_col < width:
            m[:, :, cutoff_col:] = 0.0
        left_cut = width - cutoff_col
        if trim_left and 0 < left_cut < width:
            m[:, :, :left_cut] = 0.0

        return (m,)


class SimpleImageMath:
    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "calculate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "instruction": ("STRING", {"multiline": False, "default": "img1 + img2"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            }
        }

    def calculate(self, image1, image2, instruction, threshold):
        result = eval(instruction, {"__builtins__": None}, {"img1": image1, "img2": image2})
        result = torch.clamp(torch.abs(result), 0.0, 1.0)
        rgb = result[..., :3]
        mask = (rgb > threshold).any(dim=-1).to(result.dtype)
        return (result, mask,)

class SimpleMaskMath:
    CATEGORY = "mask/transform"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "calculate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "instruction": ("STRING", {"multiline": False, "default": "m1 & m2"}),
            }
        }

    def calculate(self, mask1, mask2, instruction):
        m1 = (mask1 > 0.5).float()
        m2 = (mask2 > 0.5).float()
        result = eval(instruction, {"__builtins__": None}, {"m1": m1, "m2": m2})
        result = (result > 0.5).float()  # force final 0/1
        return (result,)


class ZoomReverse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "meta": ("DICT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "zoom_reverse"
    CATEGORY = "image/transform"

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
            img = image[i].permute(2, 0, 1).unsqueeze(0)  # BCHW
            resized_img = comfy.utils.common_upscale(
                img, crop_w, crop_h,
                upscale_method="bicubic",
                crop="disabled"
            ).squeeze(0).permute(1, 2, 0)  # HWC

            m = mask[i].unsqueeze(0).unsqueeze(0)  # BCHW
            resized_mask = comfy.utils.common_upscale(
                m, crop_w, crop_h,
                upscale_method="nearest-exact",
                crop="disabled"
            ).squeeze(0).squeeze(0)  # HW

            canvas_img = torch.zeros((orig_h, orig_w, c), dtype=torch.float32, device=image.device)
            canvas_msk = torch.zeros((orig_h, orig_w), dtype=torch.float32, device=image.device)

            canvas_img[y1:y1+crop_h, x1:x1+crop_w] = resized_img
            canvas_msk[y1:y1+crop_h, x1:x1+crop_w] = resized_mask

            restored_images.append(canvas_img)
            restored_masks.append(canvas_msk)

        restored_images = torch.stack(restored_images, dim=0)
        restored_masks = torch.stack(restored_masks, dim=0)

        return (restored_images, restored_masks)


class ZoomAlign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "p832": ("FLOAT", {"default": 1.0045, "min": 0.0, "max": 100.0, "step": 0.0001}),
                "p880": ("FLOAT", {"default": 1.003, "min": 0.0, "max": 100.0, "step": 0.0001}),
                "p1024": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.0001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "meta")
    FUNCTION = "zoom_align"
    CATEGORY = "image/transform"

    def zoom_align(self, image: torch.Tensor, p832, p880, p1024):
        b, h, w, c = image.shape

        if w == 832 or h == 832:
            zoom = p832
        elif w == 880 or h == 880:
            zoom = p880
        else:
            zoom = p1024

        if zoom == 1.0:
            meta = {"zoom": 1.0, "width": w, "height": h}
            return (image, meta)

        crop_w = max(1, int(w / zoom))
        crop_h = max(1, int(h / zoom))

        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2

        result = []
        for i in range(b):
            img = image[i].permute(2, 0, 1).unsqueeze(0)  # BCHW
            cropped = img[:, :, y1:y1+crop_h, x1:x1+crop_w]
            zoomed = comfy.utils.common_upscale(
                cropped,
                w,
                h,
                upscale_method="bicubic",
                crop="disabled"
            )

            zoomed = zoomed.squeeze(0).permute(1, 2, 0)  # HWC
            result.append(zoomed)

        result = torch.stack(result, dim=0)

        meta = {
            "zoom": zoom,
            "width": w,
            "height": h,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "x1": x1,
            "y1": y1
        }

        return (result, meta)




def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)



class FilmGrain:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "density": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "supersample_factor": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "film_grain"
    CATEGORY = "WAS Suite/Image/Filter"

    def film_grain(self, image, density, intensity, supersample_factor):
        return (pil2tensor(self.apply_film_grain(tensor2pil(image), density, intensity, supersample_factor)), )

    def apply_film_grain(self, img, density=0.1, intensity=1.0, supersample_factor=4):
        img_gray = img.convert('L')
        original_size = img.size
        img_gray = img_gray.resize(
            (img.size[0] * supersample_factor, img.size[1] * supersample_factor),
            Image.Resampling.BICUBIC
        )
        arr = np.array(img_gray, dtype=np.uint8)


        mask = (np.random.rand(*arr.shape) < density)
        arr[mask] = np.random.randint(0, 256, mask.sum(), dtype=np.uint8)
        img_noise = Image.fromarray(arr, mode='L').convert('RGB')

        img_noise = img_noise.filter(ImageFilter.GaussianBlur(radius=0.125))
        img_noise = img_noise.resize(original_size, Image.Resampling.LANCZOS)
        img_noise = img_noise.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img_final = Image.blend(img, img_noise, intensity)
        return img_final





class EstimateNoiseLevel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "estimate"
    CATEGORY = "image/noise"

    def estimate(self, image):
        img = image[0].cpu().numpy()  # (H, W, C)
        img = np.clip(img, 0, 1)
        sigma = estimate_sigma(img, channel_axis=-1, average_sigmas=True)
        return (float(sigma),)


class SumMask:
    CATEGORY = "mask/analysis"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calculate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    def calculate(self, mask):
        m = (mask > 0.5).float()
        total = torch.sum(m).item()
        return (total,)






class ImageAlignNode:
    """Advanced hybrid alignment - handles all background types"""
    CATEGORY = "image/compare"
    RETURN_TYPES = ("DICT","MASK")
    FUNCTION = "calculate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "method": (["auto", "orb", "phase", "template"], {"default": "auto"}),
            },
        }

    def _to_numpy(self, tensor):
        arr = tensor.detach().cpu().numpy()
        if arr.dtype == np.uint8: return arr
        if arr.max() <= 1.0: return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _analyze_background(self, img1, img2, bg_mask):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        bg1 = gray1[bg_mask]
        variance = np.var(bg1)
        edges1 = cv2.Canny(gray1, 50, 150)
        edge_density = edges1[bg_mask].sum() / bg_mask.sum()
        if variance < 100 or edge_density < 0.01: return 'phase'
        elif variance < 500 or edge_density < 0.05: return 'template'
        else: return 'orb'

    def _align_phase(self, img1, img2, bg_mask):
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            bg1, bg2 = gray1.copy(), gray2.copy()
            bg1[~bg_mask] = 0; bg2[~bg_mask] = 0
            shift, response = cv2.phaseCorrelate(np.float32(bg1), np.float32(bg2))
            if response < 0.3: return None
            H, W = img1.shape[:2]
            return {'scale': 1.0, 'scale_pct': 0.0, 'tx': float(shift[0]), 'ty': float(shift[1]),
                    'tx_pct': float(shift[0]/W*100), 'ty_pct': float(shift[1]/H*100),
                    'confidence': float(response), 'method': 'phase'}
        except: return None

    def _align_template(self, img1, img2, bg_mask):
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            H, W = gray1.shape
            bg_mask_uint8 = (bg_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(bg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            template_size = min(w, h, 100)
            cx, cy = x + w//2, y + h//2
            ts = template_size // 2
            if cx-ts<0 or cy-ts<0 or cx+ts>W or cy+ts>H: return None
            template = gray1[cy-ts:cy+ts, cx-ts:cx+ts]
            best_score, best_params = -1, None
            for scale in [0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05]:
                scaled = cv2.resize(gray2, None, fx=scale, fy=scale)
                if scaled.shape[0]<template.shape[0] or scaled.shape[1]<template.shape[1]: continue
                result = cv2.matchTemplate(scaled, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    found_x, found_y = max_loc
                    tx = (found_x - (int(cx*scale)-ts)) / scale
                    ty = (found_y - (int(cy*scale)-ts)) / scale
                    best_params = {'scale': scale, 'scale_pct': (scale-1.0)*100, 'tx': tx, 'ty': ty,
                                   'tx_pct': tx/W*100, 'ty_pct': ty/H*100,
                                   'confidence': float(max_val), 'method': 'template'}
            return best_params if best_score > 0.5 else None
        except: return None

    def _align_orb(self, img1, img2, bg_mask):
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            H, W = gray1.shape
            orb_mask = (bg_mask * 255).astype(np.uint8)
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(gray1, orb_mask)
            kp2, des2 = orb.detectAndCompute(gray2, orb_mask)
            if des1 is None or des2 is None or len(kp1)<10 or len(kp2)<10: return None
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m[0] for m in matches if len(m)==2 and m[0].distance < 0.75*m[1].distance]
            if len(good) < 10: return None
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC,
                                                      ransacReprojThreshold=3.0, maxIters=2000)
            if M is None: return None
            a, b, tx = M[0]; c, d, ty = M[1]
            scale = (np.sqrt(a*a + c*c) + np.sqrt(b*b + d*d)) / 2.0
            return {'scale': float(scale), 'scale_pct': float((scale-1.0)*100), 'tx': float(tx), 'ty': float(ty),
                    'tx_pct': float(tx/W*100), 'ty_pct': float(ty/H*100),
                    'confidence': float(inliers.sum()/len(good)) if inliers is not None else 0.0,
                    'matches': len(good), 'inliers': int(inliers.sum()) if inliers is not None else 0,
                    'method': 'orb'}
        except: return None

    def _select_best(self, results, img1, img2, bg_mask):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
        H, W = gray1.shape
        best_error, best_params = float('inf'), None
        for params in results:
            M = np.array([[params['scale'], 0, params['tx']], [0, params['scale'], params['ty']]], dtype=np.float32)
            warped = cv2.warpAffine(gray2, M, (W, H), flags=cv2.INTER_LINEAR)
            error = np.mean(np.abs(gray1 - warped)[bg_mask])
            if error < best_error:
                best_error, best_params = error, params
        return best_params

    def calculate(self, image1, image2, mask=None, method="auto"):
        img1 = self._to_numpy(image1[0])
        img2 = self._to_numpy(image2[0])
        H, W = img1.shape[:2]
        
        if mask is not None:
            mask_np = self._to_numpy(mask[0])
            if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
            bg_mask = mask_np < 128
        else:
            bg_mask = np.ones((H, W), dtype=bool)
        
        results = []
        if method == "auto":
            selected = self._analyze_background(img1, img2, bg_mask)
            for m in [selected, 'orb', 'phase', 'template']:
                if m == 'phase': res = self._align_phase(img1, img2, bg_mask)
                elif m == 'template': res = self._align_template(img1, img2, bg_mask)
                else: res = self._align_orb(img1, img2, bg_mask)
                if res is not None and res not in results: results.append(res)
        else:
            if method == 'phase': res = self._align_phase(img1, img2, bg_mask)
            elif method == 'template': res = self._align_template(img1, img2, bg_mask)
            else: res = self._align_orb(img1, img2, bg_mask)
            if res is not None: results.append(res)
        
        if not results: return ({"error": "alignment_failed"}, None)
        
        best = self._select_best(results, img1, img2, bg_mask) if len(results) > 1 else results[0]
        
        M = np.array([[best['scale'], 0, best['tx']], [0, best['scale'], best['ty']]], dtype=np.float32)
        aligned = cv2.warpAffine(img2, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        diff = np.abs(img1.astype(float) - aligned.astype(float)).mean(axis=2) / 255.0
        diff_mask = (diff > 0.01).astype(np.float32)
        diff_mask[~bg_mask] = 0.0
        
        return (best, torch.from_numpy(diff_mask).unsqueeze(0))



class ApplyImageAlign:
    CATEGORY = "image/compare"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "calculate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "info": ("DICT",),
            }
        }

    def _to_numpy(self, tensor):
        arr = tensor.detach().cpu().numpy()
        if arr.dtype == np.uint8: return arr
        if arr.max() <= 1.0: return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def calculate(self, image1, image2, info):
        img2_np = self._to_numpy(image2[0])
        H, W = img2_np.shape[:2]
        scale = 1.0 + (info.get("scale_pct", 0.0) / 100.0)
        tx = info.get("tx_pct", 0.0) / 100.0 * W
        ty = info.get("ty_pct", 0.0) / 100.0 * H
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
        aligned = cv2.warpAffine(img2_np, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        out = torch.from_numpy(aligned.astype(np.float32) / 255.0)
        out = out.permute(2, 0, 1).unsqueeze(0).permute(0, 2, 3, 1)
        return (out,)



class FilterMaskRegions:
    CATEGORY = "mask/transform"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "max_regions": ("INT", {"default": 3, "min": 0, "max": 1000}), "min_area": ("INT", {"default": 1, "min": 0, "max": 10000000})}}

    def process(self, mask, max_regions=3, min_area=1):
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor with shape [B,H,W]")
        if mask.ndim != 3:
            raise ValueError("mask must have shape [B,H,W]")
        arr = mask[0].detach().cpu().numpy()
        bw = (arr > 0.5).astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if n <= 1 or int(max_regions) <= 0:
            out = np.zeros_like(bw, dtype=np.float32)
            return (torch.from_numpy(out).unsqueeze(0),)
        areas = stats[1:, cv2.CC_STAT_AREA]
        idxs = np.arange(1, n)[areas >= int(min_area)]
        if idxs.size == 0:
            out = np.zeros_like(bw, dtype=np.float32)
            return (torch.from_numpy(out).unsqueeze(0),)
        order = np.argsort(-areas[areas >= int(min_area)])
        keep = idxs[order][:int(max_regions)]
        out = np.zeros_like(bw, dtype=np.uint8)
        for lbl in keep:
            out[labels == int(lbl)] = 1
        out_t = torch.from_numpy(out.astype(np.float32)).unsqueeze(0)
        return (out_t,)



class GrowMaskPct:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "expand_pct": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}), "tapered_corners": ("BOOLEAN", {"default": True})}}
    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand_mask"

    def expand_mask(self, mask, expand_pct=0.0, tapered_corners=True):
        kernel = np.array([[0 if tapered_corners else 1, 1, 0 if tapered_corners else 1],
                           [1, 1, 1],
                           [0 if tapered_corners else 1, 1, 0 if tapered_corners else 1]])
        masks = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in masks:
            arr = m.numpy()
            H, W = arr.shape[-2], arr.shape[-1]
            steps = int(round(abs(float(expand_pct)) * max(H, W) / 100.0))
            if steps == 0:
                out.append(torch.from_numpy(arr))
                continue
            for _ in range(steps):
                if expand_pct < 0:
                    arr = scipy.ndimage.grey_erosion(arr, footprint=kernel)
                else:
                    arr = scipy.ndimage.grey_dilation(arr, footprint=kernel)
            out.append(torch.from_numpy(arr))
        return (torch.stack(out, dim=0),)



class CropSubject:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "top_margin_pct": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "bottom_cut_pct": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "side_margin_pct": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_subject"
    CATEGORY = "Image Processing"

    def crop_subject(self, image, mask, top_margin_pct, bottom_cut_pct, side_margin_pct):
        img = image[0]
        mask_arr = 1 - mask[0]
        coords = torch.nonzero(mask_arr > 0, as_tuple=False)
        if coords.numel() == 0:
            return (image,)
        ys = coords[:, 0]; xs = coords[:, 1]
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        subj_h = y_max - y_min
        top_expand = int(subj_h * float(top_margin_pct))
        bottom_crop = int(subj_h * float(bottom_cut_pct))
        y0 = max(0, y_min - top_expand)
        y1 = max(y0 + 1, y_max - bottom_crop)
        cropped_v = img[y0:y1, :, :]
        Hc, Wc, C = cropped_v.shape
        subj_w = x_max - x_min
        margin = int(subj_w * float(side_margin_pct))
        x0 = max(0, x_min - margin)
        x1 = min(img.shape[1], x_max + margin)
        x1 = max(x0 + 1, x1)
        cropped = cropped_v[:, x0:x1, :]
        out = cropped.unsqueeze(0)
        return (out,)



NODE_CLASS_MAPPINGS = {
    "ZyCropByMaskBBox": CropByMaskBBox,
    "ZyStitchNode": StitchNode,
    "ZyTrimImage": TrimImage,
    "ZyMaskModifier": MaskModifier,
    "ZySimpleImageMath": SimpleImageMath,
    "ZySimpleMaskMath": SimpleMaskMath,
    "ZyZoomReverse": ZoomReverse,
    "ZyZoomAlign": ZoomAlign,
    "ZyFilmGrain": FilmGrain,
    "ZyEstimateNoiseLevel": EstimateNoiseLevel,
    "ZySumMask": SumMask,
    "ZyImageAlignNode": ImageAlignNode,
    "ZyApplyImageAlign": ApplyImageAlign,
    "ZyFilterMaskRegions": FilterMaskRegions,
    "ZyGrowMaskPct": GrowMaskPct,
    "ZyCropSubject": CropSubject,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZyCropByMaskBBox": "Zy CropByMaskBBox",
    "ZyStitchNode": "Zy StitchNode",
    "ZyTrimImage": "Zy TrimImage",
    "ZyMaskModifier": "Zy MaskModifier",
    "ZySimpleImageMath": "Zy SimpleImageMath",
    "ZySimpleMaskMath": "Zy SimpleMaskMath",
    "ZyZoomReverse": "Zy ZoomReverse",
    "ZyZoomAlign": "Zy ZoomAlign",
    "ZyFilmGrain": "Zy FilmGrain",
    "ZyEstimateNoiseLevel": "Zy EstimateNoiseLevel",
    "ZySumMask": "Zy SumMask",
    "ZyImageAlignNode": "Zy ImageAlignNode",
    "ZyApplyImageAlign": "Zy ApplyImageAlign",
    "ZyFilterMaskRegions": "Zy FilterMaskRegions",
    "ZyGrowMaskPct": "Zy GrowMaskPct",
    "ZyCropSubject": "Zy CropSubject",
}
