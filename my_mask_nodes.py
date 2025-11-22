"""
My Mask Processing Nodes - Complex mask crop and blending operations
Separated due to size and complexity
"""

import comfy.utils
import cv2
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

from .zy_nodes import fast_blur



class MyMaskCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "mask": ("MASK",),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01
                }),
                "width": ("INT", {
                    "default": 832, 
                    "min": 64, #Minimum value
                    "max": 1216, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "height": ("INT", {
                    "default": 1216, 
                    "min": 64, #Minimum value
                    "max": 1216, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                })
            },
        }

    RETURN_TYPES = ("IMAGE","MASK","LIST",)
    FUNCTION = "crop_images"
    CATEGORY = "Custom"

    def crop_images(self, image1, mask, scale_factor, width, height):
        TOP_MARGIN_PERCENT = 0.095
        BOTTOM_MARGIN_PERCENT = 0.04
        SIDE_MARGIN_PERCENT = 0.01
        DESIRED_WIDTH = width
        DESIRED_HEIGHT = height
        DESIRED_ASPECT_RATIO = DESIRED_HEIGHT / DESIRED_WIDTH
        
        image = image1[0].numpy()
        mask = mask[0].numpy()
        coords = np.column_stack(np.where(mask > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        height = y_max - y_min + 1
        width = x_max - x_min + 1

        factor = width

        if height > 1.4*width:
            factor = height * 0.28

        factor = factor * scale_factor
        
        top_margin = int(TOP_MARGIN_PERCENT * factor)
        bottom_margin = int(BOTTOM_MARGIN_PERCENT * factor)
        side_margin = int(SIDE_MARGIN_PERCENT * factor)

        y_min = max(0, y_min - top_margin)
        y_max = min(image.shape[0] - 1, y_max + bottom_margin)
        x_min = max(0, x_min - side_margin)
        x_max = min(image.shape[1] - 1, x_max + side_margin)

        height = y_max - y_min + 1
        width = x_max - x_min + 1
        current_aspect_ratio = height / width

        if current_aspect_ratio < DESIRED_ASPECT_RATIO:
            new_height = int(width * DESIRED_ASPECT_RATIO)
            top_margin = (new_height - height) // 2
            bottom_margin = new_height - height - top_margin
            y_min = max(0, y_min - top_margin)
            y_max = min(image.shape[0] - 1, y_max + bottom_margin)
            height = y_max - y_min + 1
        else:
            new_width = int(height / DESIRED_ASPECT_RATIO)
            side_margin = (new_width - width) // 2
            x_min = max(0, x_min - side_margin)
            x_max = min(image.shape[1] - 1, x_max + side_margin)
            width = x_max - x_min + 1


        if height / width > DESIRED_ASPECT_RATIO:
            new_width = int(height / DESIRED_ASPECT_RATIO)
            x_center = (x_min + x_max) // 2
            x_min = max(0, x_center - new_width // 2)
            x_max = min(image.shape[1] - 1, x_center + new_width // 2)
        else:
            new_height = int(width * DESIRED_ASPECT_RATIO)
            y_center = (y_min + y_max) // 2
            y_min = max(0, y_center - new_height // 2)
            y_max = min(image.shape[0] - 1, y_center + new_height // 2)
        

        TARGET_AREA = DESIRED_WIDTH * DESIRED_HEIGHT
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        current_area = width * height
        scaling_factor = (TARGET_AREA / current_area) ** 0.5
        scaled_width = int(width * scaling_factor)
        scaled_height = int(height * scaling_factor)

        DESIRED_WIDTH = (scaled_width + 31) // 64 * 64
        DESIRED_HEIGHT = (scaled_height + 31) // 64 * 64

        print(f"DESIRED_WIDTH: {DESIRED_WIDTH}, DESIRED_HEIGHT: {DESIRED_HEIGHT}")

        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        resized_image = cv2.resize(cropped_image, (DESIRED_WIDTH, DESIRED_HEIGHT))
        resized_image_tensor = torch.tensor(resized_image).unsqueeze(0)

        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        resized_mask = cv2.resize(cropped_mask, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_NEAREST)
        resized_mask_tensor = torch.tensor(resized_mask).unsqueeze(0)

        crop_coords = [y_min, y_max + 1, x_min, x_max + 1]
        
        return (resized_image_tensor,resized_mask_tensor,crop_coords,)




NODE_CLASS_MAPPINGS = {
    "MyMaskCrop": MyMaskCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyMaskCrop": "My Mask Crop"
}


class MyMaskCropGrow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "mask": ("MASK",),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.20,
                    "max": 5.0,
                    "step": 0.05
                }),
                "width": ("INT", {
                    "default": 832, 
                    "min": 64, #Minimum value
                    "max": 1216, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "height": ("INT", {
                    "default": 1216, 
                    "min": 64, #Minimum value
                    "max": 1216, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "left": ("INT", {"default": 0, "min": 0, "max": 1216, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": 1216, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": 1216, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 1216, "step": 8}),
                "feathering": ("INT", {"default": 15, "min": 0, "max": 1216, "step": 5}),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK","LIST","IMAGE","MASK",)
    FUNCTION = "crop_images"
    CATEGORY = "Custom"

    def crop_images(self, image1, mask, scale_factor, width, height, left, top, right, bottom, feathering):
        ext_image, ext_mask = self.expand_image(image1, mask, left, top, right, bottom, feathering)
        ext_mask, = self.expand_mask(ext_mask, feathering, True)

        TOP_MARGIN_PERCENT = 0.2
        BOTTOM_MARGIN_PERCENT = 0.1
        SIDE_MARGIN_PERCENT = 0.05
        DESIRED_WIDTH = width
        DESIRED_HEIGHT = height
        DESIRED_ASPECT_RATIO = DESIRED_HEIGHT / DESIRED_WIDTH
        
        image = ext_image[0].numpy()
        mask = ext_mask[0].numpy()
        coords = np.column_stack(np.where(mask > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        height = y_max - y_min + 1
        width = x_max - x_min + 1

        factor = width

        if height > 1.4*width:
            factor = height * 0.28

        factor = factor * scale_factor
        
        top_margin = int(TOP_MARGIN_PERCENT * factor)
        bottom_margin = int(BOTTOM_MARGIN_PERCENT * factor)
        side_margin = int(SIDE_MARGIN_PERCENT * factor)

        y_min = max(0, y_min - top_margin)
        y_max = min(image.shape[0] - 1, y_max + bottom_margin)
        x_min = max(0, x_min - side_margin)
        x_max = min(image.shape[1] - 1, x_max + side_margin)

        height = y_max - y_min + 1
        width = x_max - x_min + 1
        current_aspect_ratio = height / width

        if current_aspect_ratio < DESIRED_ASPECT_RATIO:
            new_height = int(width * DESIRED_ASPECT_RATIO)
            top_margin = (new_height - height) // 2
            bottom_margin = new_height - height - top_margin
            y_min = max(0, y_min - top_margin)
            y_max = min(image.shape[0] - 1, y_max + bottom_margin)
            height = y_max - y_min + 1
        else:
            new_width = int(height / DESIRED_ASPECT_RATIO)
            side_margin = (new_width - width) // 2
            x_min = max(0, x_min - side_margin)
            x_max = min(image.shape[1] - 1, x_max + side_margin)
            width = x_max - x_min + 1


        if height / width > DESIRED_ASPECT_RATIO:
            new_width = int(height / DESIRED_ASPECT_RATIO)
            x_center = (x_min + x_max) // 2
            x_min = max(0, x_center - new_width // 2)
            x_max = min(image.shape[1] - 1, x_center + new_width // 2)
        else:
            new_height = int(width * DESIRED_ASPECT_RATIO)
            y_center = (y_min + y_max) // 2
            y_min = max(0, y_center - new_height // 2)
            y_max = min(image.shape[0] - 1, y_center + new_height // 2)
        

        TARGET_AREA = DESIRED_WIDTH * DESIRED_HEIGHT
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        current_area = width * height
        scaling_factor = (TARGET_AREA / current_area) ** 0.5
        scaled_width = int(width * scaling_factor)
        scaled_height = int(height * scaling_factor)

        DESIRED_WIDTH = (scaled_width + 31) // 64 * 64
        DESIRED_HEIGHT = (scaled_height + 31) // 64 * 64

        print(f"DESIRED_WIDTH: {DESIRED_WIDTH}, DESIRED_HEIGHT: {DESIRED_HEIGHT}")

        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        resized_image = cv2.resize(cropped_image, (DESIRED_WIDTH, DESIRED_HEIGHT))
        resized_image_tensor = torch.tensor(resized_image).unsqueeze(0)

        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        resized_mask = cv2.resize(cropped_mask, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_NEAREST)
        resized_mask_tensor = torch.tensor(resized_mask).unsqueeze(0)

        crop_coords = [y_min, y_max + 1, x_min, x_max + 1, resized_image_tensor, resized_mask_tensor, ext_image, ext_mask]
        
        return (resized_image_tensor,resized_mask_tensor,crop_coords,ext_image,ext_mask,)


    def expand_image(self, image, inputmask, left, top, right, bottom, feathering):
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + d2, left:left + d3, :] = image

        new_mask = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        new_mask[:, top:top + d2, left:left + d3] = inputmask
        mask = new_mask

        return (new_image, mask)

    def expand_mask(self, mask, expand, tapered_corners):
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        return (torch.stack(out, dim=0),)




NODE_CLASS_MAPPINGS = {
    "MyMaskCropGrow": MyMaskCropGrow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyMaskCropGrow": "My Mask Crop Grow"
}


class MyMaskCropJoin:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "pos": ("LIST",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "join_images"
    CATEGORY = "Custom"

    def join_images(self, image1, image2, pos):
        y_min, y_max, x_min, x_max = pos
        target_height = y_max - y_min
        target_width = x_max - x_min
        image2_new = image2.permute(0, 3, 1, 2)
        scaled_images = F.interpolate(image2_new, size=(target_height, target_width), mode='bilinear', align_corners=False)
        image1_new = image1.clone()
        image1_new[0, y_min:y_max, x_min:x_max] = scaled_images.permute(0,2,3,1)
        return (image1_new,)

NODE_CLASS_MAPPINGS = {
    "MyMaskCropJoin": MyMaskCropJoin
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyMaskCropJoin": "My Mask Crop Join"
}


class MyMaskCropJoinBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image2": ("IMAGE",),
                "pos": ("LIST",),
                "radius": ("INT", {
                    "default": 5, 
                    "min": 5, #Minimum value
                    "max": 200, #Maximum value
                    "step": 4, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "join_images"
    CATEGORY = "Custom"

    def join_images(self, image2, pos, radius):
        y_min, y_max, x_min, x_max, rez_image, rez_mask, ext_image, ext_mask = pos
        brightness_diff = self.compute_brightness_diff(rez_image, image2, rez_mask)
        print(f"brightness_diff: {brightness_diff.tolist()}")
        image2_adjusted = self.adjust_brightness(image2, brightness_diff / 2.0)
        target_height = y_max - y_min
        target_width = x_max - x_min
        scaled_images = F.interpolate(image2_adjusted.permute(0, 3, 1, 2), size=(target_height, target_width), mode='bilinear', align_corners=False)
        image1_new = ext_image.clone()
        image1_new[0, y_min:y_max, x_min:x_max] = scaled_images.permute(0,2,3,1)
        blended_image, = self.blend_images(ext_image, image1_new, ext_mask, radius)
        return (blended_image,)

    def blend_images(self, image1, image2, mask, radius):
        blur_radius = radius
        blurred_mask = TVF.gaussian_blur(mask.float(), kernel_size=blur_radius)
        mask = blurred_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        blended_image = image1 * (1 - mask) + image2 * mask
        return (blended_image,)

    # def compute_and_adjust_brightness_per_channel(self, image1, image2, mask):
    #     diff = image1[new_mask.bool()].mean(dim=0) - image2[new_mask.bool()].mean(dim=0)

    def compute_brightness_diff(self, image1, image2, mask):
        new_mask = 1 - mask
        avg_brightness1 = image1[new_mask.bool()].mean(dim=0)
        avg_brightness2 = image2[new_mask.bool()].mean(dim=0)
        return avg_brightness1 - avg_brightness2

    def adjust_brightness(self, image, diff):
        return torch.clamp(image + diff, 0, 1)

NODE_CLASS_MAPPINGS = {
    "MyMaskCropJoinBlur": MyMaskCropJoinBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyMaskCropJoinBlur": "My Mask Crop Join Blur"
}


class MyMaskImageBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "mask": ("MASK",),
                "radius": ("INT", {
                    "default": 5, 
                    "min": 5,
                    "max": 200,
                    "step": 2,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"
    CATEGORY = "Custom"

    def blend_images(self, image1, image2, mask, radius):
        blur_radius = radius
        blurred_mask = TVF.gaussian_blur(mask.float(), kernel_size=blur_radius)
        mask = blurred_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        blended_image = image1 * (1 - mask) + image2 * mask
        return (blended_image,)


NODE_CLASS_MAPPINGS = {
    "MyMaskImageBlender": MyMaskImageBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyMaskImageBlender": "My Mask Image Blender"
}




class MySimpleImageMath:
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

class MyMaskModifier:
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
        trim_top    = bool(flags & (1 << 4))

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

class MyTrimImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "meta": ("TENSOR",),
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

    def trim_image(self, image: torch.Tensor, meta, trim_percent: float):
        y0, x0, crop_h, crop_w, pad_top, pad_left, target_h, target_w = [int(x) for x in meta.tolist()]
        _, h, w, c = image.shape

        if target_h == 1024:
            return (image, 88,)

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

        intvl = (int(not trim_bottom) << 0) | (int(not trim_left) << 1) | (int(not trim_right) << 2) | (int(not trim_top) << 4)

        return (trimmed, intvl,)



class MyDynamicCropNode:
    CANDIDATES=[(1024,1024),(832,1248),(880,1184),(1184,880),(1248,832)]
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"image":("IMAGE",),"mask":("MASK",),"margin_pct":("FLOAT",{"default":0.1,"min":0,"max":0.5,"step":0.01})}}
    RETURN_TYPES=("IMAGE","INT","INT","TENSOR"); RETURN_NAMES=("patch","height","width","meta"); FUNCTION="crop_subject"; CATEGORY="Image Processing"

    def _choose_target(self,bh,bw,H):
        r_bbox = bh/bw if bw>0 else 1.0
        best=None; best_diff=1e9
        for th,tw in self.CANDIDATES:
            r = th/tw
            crop_h = bw * r
            if crop_h <= H:
                diff = abs(r - r_bbox)
                if diff < best_diff:
                    best=(th,tw); best_diff=diff
        if best is not None:
            return best
        best=None; best_over=1e18; best_diff=1e18
        for th,tw in self.CANDIDATES:
            r = th/tw
            crop_h = bw * r
            over = max(0, crop_h - H)
            diff = abs(r - r_bbox)
            if over < best_over or (over==best_over and diff < best_diff):
                best=(th,tw); best_over=over; best_diff=diff
        return best

    def crop_subject(self,image,mask,margin_pct):
        img=image[0]; H,W,C=img.shape
        coords=torch.nonzero((1-mask[0])>0, as_tuple=False)
        if coords.numel()==0:
            th,tw=1024,1024
            crop_w=min(W,tw); crop_h=min(H,th)
            x0=max(0,(W-crop_w)//2); y0=max(0,H-crop_h)
        else:
            ys,xs=coords[:,0],coords[:,1]
            y_min,y_max=int(ys.min()),int(ys.max()); x_min,x_max=int(xs.min()),int(xs.max())
            bh,bw = max(1,y_max-y_min+1), max(1,x_max-x_min+1)
            eh,ew = int(round(bh*margin_pct)), int(round(bw*margin_pct))
            y0b,y1b = max(0, y_min - eh//2), min(H, y_max + (eh - eh//2))
            x0b,x1b = max(0, x_min - ew//2), min(W, x_max + (ew - ew//2))
            bh_exp, bw_exp = max(1, y1b-y0b), max(1, x1b-x0b)
            th,tw = self._choose_target(bh_exp, bw_exp, H)
            r = th/tw
            crop_w = min(W, bw_exp)
            crop_h = max(1, int(round(crop_w * r)))
            if crop_h > H:
                scale = H / float(crop_h)
                crop_h = H
                crop_w = max(1, int(round(crop_w * scale)))
            cx = (x0b + x1b)//2
            x0 = int(max(0, min(cx - crop_w//2, W - crop_w)))
            y0 = int(max(0, H - crop_h))
        piece = img[int(y0):int(y0+crop_h), int(x0):int(x0+crop_w), :].permute(2,0,1).unsqueeze(0)
        if piece.shape[2]==0 or piece.shape[3]==0:
            piece = img[max(0,H-1):H, max(0,W-1):W, :].permute(2,0,1).unsqueeze(0)
        patch_bchw = F.interpolate(piece, size=(th,tw), mode="bilinear", align_corners=False)
        patch_hwc = patch_bchw[0].permute(1,2,0)
        meta = torch.tensor([int(y0), int(x0), int(crop_h), int(crop_w), 0, 0, int(th), int(tw)], dtype=torch.int32)
        return (patch_hwc.unsqueeze(0), int(th), int(tw), meta)



class MyDynamicStitchNode:
    """
    Expects meta format: [y0, x0, crop_h, crop_w, pad_top, pad_left, target_h, target_w]
    - Resize replacement patch back to (crop_h, crop_w)
    - Paste overlapping region into original image at (y0..y0+crop_h, x0..x0+crop_w)
    - Handles cases where crop exceeded original image bounds (uses pad offsets to compute overlap)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_image": ("IMAGE",),
            "replacement": ("IMAGE",),
            "meta": ("TENSOR",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch_back"
    CATEGORY = "Image Processing"

    def stitch_back(self, original_image, replacement, meta):
        img = original_image[0].clone()  # H x W x C
        rep = replacement[0]             # target_h x target_w x C
        y0, x0, crop_h, crop_w, pad_top, pad_left, target_h, target_w = [int(x) for x in meta.tolist()]
        H, W, C = img.shape

        rep_bchw = F.interpolate(rep.permute(2, 0, 1).unsqueeze(0), size=(crop_h, crop_w),
                                 mode="bilinear", align_corners=False)[0]
        rep_hwc = rep_bchw.permute(1, 2, 0)

        # compute region of rep that corresponds to the original image
        src_y0 = pad_top
        src_x0 = pad_left
        src_y1 = pad_top + min(crop_h - pad_top, max(0, H - max(0, y0)))
        src_x1 = pad_left + min(crop_w - pad_left, max(0, W - max(0, x0)))

        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 <= src_y0 or src_x1 <= src_x0 or dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            return (img.unsqueeze(0),)

        src_patch = rep_hwc[src_y0:src_y1, src_x0:src_x1, :]
        img[dst_y0:dst_y1, dst_x0:dst_x1, :] = src_patch

        return (img.unsqueeze(0),)



class MyDynamicStitchMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_image": ("IMAGE",),
            "replacement": ("IMAGE",),
            "meta": ("TENSOR",),       # [y0, x0, crop_h, crop_w, pad_top, pad_left, target_h, target_w]
            "mask": ("MASK",),         # same spatial size as 'replacement' (target_h, target_w)
            "blur_radius": ("INT", {"default": 5, "min": 0, "max": 128}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch_with_mask"
    CATEGORY = "Image Processing"

    def stitch_with_mask(self, original_image, replacement, meta, mask, blur_radius):
        orig = original_image[0].clone()    # H x W x C
        rep = replacement[0]                # target_h x target_w x C
        msk = mask[0]                       # target_h x target_w (single channel)

        H_img, W_img, C = orig.shape

        arr = [int(x) for x in meta.cpu().tolist()]
        if len(arr) != 8:
            return (orig.unsqueeze(0),)
        y0, x0, crop_h, crop_w, pad_top, pad_left, target_h, target_w = arr

        if crop_h <= 0 or crop_w <= 0:
            return (orig.unsqueeze(0),)

        y1 = y0 + crop_h
        x1 = x0 + crop_w

        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = min(H_img, y1)
        dst_x1 = min(W_img, x1)
        slice_h = dst_y1 - dst_y0
        slice_w = dst_x1 - dst_x0
        if slice_h <= 0 or slice_w <= 0:
            return (orig.unsqueeze(0),)

        # where in the resized crop the overlapping region starts (account for padding/offset)
        src_y0 = pad_top + (dst_y0 - y0)
        src_x0 = pad_left + (dst_x0 - x0)
        src_y1 = src_y0 + slice_h
        src_x1 = src_x0 + slice_w

        try:
            rep_bchw = F.interpolate(
                rep.permute(2, 0, 1).unsqueeze(0).float(),
                size=(crop_h, crop_w),
                mode="bilinear", align_corners=False
            )[0]
        except Exception:
            return (orig.unsqueeze(0),)
        rep_resized = rep_bchw.permute(1, 2, 0).to(orig.dtype)

        try:
            msk_b = F.interpolate(
                msk.unsqueeze(0).unsqueeze(0).float(),
                size=(crop_h, crop_w),
                mode="bilinear", align_corners=False
            )[0, 0]
        except Exception:
            return (orig.unsqueeze(0),)

        # blur mask if requested (use cv2 for reliable odd-kernel blur)
        if blur_radius and blur_radius > 0:
            k = int(max(1, 2 * int(blur_radius) + 1))
            m_np = msk_b.cpu().numpy()
            if m_np.size > 0:
                m_blur = cv2.GaussianBlur(m_np, (k, k), 0)
                msk_b = torch.from_numpy(m_blur).to(msk_b.device).float()

        msk_b = msk_b.clamp(0.0, 1.0).unsqueeze(-1)   # Hcrop x Wcrop x 1

        # slice source patch and mask patch (these correspond to the overlapping part)
        src_patch = rep_resized[src_y0:src_y1, src_x0:src_x1, :]
        src_mask_patch = msk_b[src_y0:src_y1, src_x0:src_x1, :]

        # Safety: if sizes mismatch (rare), resize to match slice dims
        if src_patch.shape[0] != slice_h or src_patch.shape[1] != slice_w:
            src_patch = F.interpolate(
                src_patch.permute(2, 0, 1).unsqueeze(0),
                size=(slice_h, slice_w), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0)
        if src_mask_patch.shape[0] != slice_h or src_mask_patch.shape[1] != slice_w:
            src_mask_patch = F.interpolate(
                src_mask_patch.permute(2, 0, 1).unsqueeze(0),
                size=(slice_h, slice_w), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0)

        dst_region = orig[dst_y0:dst_y1, dst_x0:dst_x1, :].to(src_patch.dtype)
        blended = dst_region * (1.0 - src_mask_patch) + src_patch * src_mask_patch
        orig[dst_y0:dst_y1, dst_x0:dst_x1, :] = blended

        return (orig.unsqueeze(0),)




class MyCropSubjectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "top_margin_pct": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "bottom_cut_pct": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_subject"
    CATEGORY = "Image Processing"

    def crop_subject(self, image, mask, top_margin_pct, bottom_cut_pct):
        img = image[0]        # H x W x C
        mask_arr = mask[0]    # H x W
        mask_arr = 1 - mask_arr
        coords = torch.nonzero(mask_arr > 0, as_tuple=False)
        if coords.numel() == 0:
            return (image,)

        ys = coords[:, 0]
        xs = coords[:, 1]
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        subj_h = y_max - y_min
        top_expand = int(subj_h * top_margin_pct)
        bottom_crop = int(subj_h * bottom_cut_pct)

        y0 = max(0, y_min - top_expand)
        y1 = max(y0 + 1, y_max - bottom_crop)
        cropped_v = img[y0:y1, :, :]

        Hc, Wc, C = cropped_v.shape
        cx = (x_min + x_max) // 2
        half = min(cx, Wc - cx)
        x0 = cx - half
        x1 = cx + half
        cropped = cropped_v[:, x0:x1, :]

        out = cropped.unsqueeze(0)
        return (out,)

NODE_CLASS_MAPPINGS = {
    "MyCropSubjectNode": MyCropSubjectNode,
    "MyDynamicCropNode": MyDynamicCropNode,
    "MyDynamicStitchNode": MyDynamicStitchNode,
    "MyDynamicStitchMaskNode": MyDynamicStitchMaskNode,
    "MyTrimImage": MyTrimImage,
    "MyMaskModifier": MyMaskModifier,
    "MySimpleImageMath": MySimpleImageMath
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCropSubjectNode": "My Crop Around Subject",
    "MyDynamicCropNode": "My Dynamic Crop Around Subject",
    "MyDynamicStitchNode": "My Dynamic Stitch Back",
    "MyDynamicStitchMaskNode": "My Dynamic Stitch Mask",
    "MyTrimImage": "My Trim Image",
    "MyMaskModifier": "My Mask Modifier",
    "MySimpleImageMath": "My Simple Image Math"
}
