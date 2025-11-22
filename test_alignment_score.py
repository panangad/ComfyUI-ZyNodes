#!/usr/bin/env python3
"""
Alignment Test Script - Scores alignment quality on datasets
Usage: python test_alignment_score.py [dataset_path]
"""

import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Mock comfy module for standalone testing
if 'comfy' not in sys.modules:
    sys.modules['comfy'] = type(sys)('comfy')
    sys.modules['comfy.utils'] = type(sys)('comfy.utils')

sys.path.insert(0, str(Path(__file__).parent))
from zy_nodes import ImageAlignNode, ApplyImageAlign

def load_tensor(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def load_mask(path):
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def calc_metrics(img1, img2, mask=None):
    if isinstance(img1, torch.Tensor):
        img1 = (img1[0].cpu().numpy() * 255).astype(np.uint8)
    if isinstance(img2, torch.Tensor):
        img2 = (img2[0].cpu().numpy() * 255).astype(np.uint8)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    if mask is not None:
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        comp_mask = (mask_np < 128)
    else:
        comp_mask = np.ones_like(gray1, dtype=bool)
    
    diff = np.abs(gray1.astype(float) - gray2.astype(float))
    mae = float(np.mean(diff[comp_mask]))
    
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(gray1, gray2, data_range=255)
    except:
        ssim_val = 0.0
    
    mse = np.mean(np.square(diff[comp_mask]))
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
    
    aligned_pixels = (diff[comp_mask] < 10).sum() / comp_mask.sum() * 100
    
    return {'mae': mae, 'ssim': ssim_val, 'psnr': psnr, 'aligned_pct': aligned_pixels}

def score_alignment(mae, ssim, psnr, aligned_pct):
    ssim_score = ssim * 25
    psnr_score = min(25, max(0, (psnr - 20) / 30 * 25))
    aligned_score = (aligned_pct / 100) * 25
    edge_score = max(0, 25 - (mae / 10))
    total = ssim_score + psnr_score + aligned_score + edge_score
    
    if total >= 90: grade = 'A+'
    elif total >= 80: grade = 'A'
    elif total >= 70: grade = 'B'
    elif total >= 60: grade = 'C'
    else: grade = 'D'
    
    return total, grade

def test_dataset(ds_path):
    print(f"\n{'='*70}")
    print(f"Testing: {ds_path.name}")
    print(f"{'='*70}")
    
    img1_path = ds_path / "img1.png"
    img2_paths = [ds_path / "img2.png", ds_path / "im2.png"]
    img2_path = next((p for p in img2_paths if p.exists()), None)
    mask_path = ds_path / "mask-img.png"
    
    if not img1_path.exists() or not img2_path:
        print("❌ Missing images")
        return None
    
    img1 = load_tensor(img1_path)
    img2 = load_tensor(img2_path)
    mask = load_mask(mask_path) if mask_path.exists() else None
    
    print(f"\n📁 Images: {img1.shape[1]}x{img1.shape[2]}")
    
    aligner = ImageAlignNode()
    applier = ApplyImageAlign()
    
    try:
        import time
        start = time.time()
        info, diff_mask = aligner.calculate(img1, img2, mask=mask, method="auto")
        
        if 'error' in info:
            print(f"\n❌ Alignment failed: {info}")
            return None
        
        aligned = applier.calculate(img1, img2, info)[0]
        elapsed = time.time() - start
        
        metrics_before = calc_metrics(img1, img2, mask)
        metrics_after = calc_metrics(img1, aligned, mask)
        
        score, grade = score_alignment(metrics_after['mae'], metrics_after['ssim'], 
                                       metrics_after['psnr'], metrics_after['aligned_pct'])
        
        print(f"\n✅ Aligned in {elapsed:.3f}s")
        print(f"\nMethod: {info.get('method', 'unknown')}")
        print(f"Transform: Scale {info['scale_pct']:+.2f}%, TX {info['tx_pct']:+.2f}%, TY {info['ty_pct']:+.2f}%")
        
        print(f"\nBefore Alignment:")
        print(f"  MAE: {metrics_before['mae']:.2f}")
        print(f"  SSIM: {metrics_before['ssim']:.4f}")
        print(f"  PSNR: {metrics_before['psnr']:.2f} dB")
        
        print(f"\nAfter Alignment:")
        print(f"  MAE: {metrics_after['mae']:.2f}")
        print(f"  SSIM: {metrics_after['ssim']:.4f}")
        print(f"  PSNR: {metrics_after['psnr']:.2f} dB")
        print(f"  Aligned Pixels: {metrics_after['aligned_pct']:.1f}%")
        
        print(f"\n🎯 Score: {score:.1f}/100 [{grade}]")
        
        return {
            'name': ds_path.name,
            'method': info.get('method'),
            'time': elapsed,
            'score': score,
            'grade': grade,
            'metrics': metrics_after,
            'improvement': ((metrics_before['mae'] - metrics_after['mae']) / metrics_before['mae'] * 100)
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "="*70)
    print("IMAGE ALIGNMENT TEST & SCORING")
    print("="*70)
    
    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1])
        if test_path.exists():
            result = test_dataset(test_path)
            return 0 if result else 1
    
    base_path = Path("/mnt/ramdisk/sample-input")
    if not base_path.exists():
        base_path = Path("./sample-input")
    
    if not base_path.exists():
        print("\n❌ No test data found")
        print("Usage: python test_alignment_score.py [dataset_path]")
        return 1
    
    results = []
    for ds_path in sorted(base_path.iterdir()):
        if ds_path.is_dir():
            result = test_dataset(ds_path)
            if result:
                results.append(result)
    
    if not results:
        print("\n❌ No successful tests")
        return 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Dataset':<12} {'Method':<10} {'Score':<10} {'Grade':<8} {'Improvement':<12} {'Time'}")
    print("-"*70)
    
    total_score = 0
    for r in results:
        print(f"{r['name']:<12} {r['method']:<10} {r['score']:<10.1f} {r['grade']:<8} "
              f"{r['improvement']:<12.1f}% {r['time']:.3f}s")
        total_score += r['score']
    
    avg_score = total_score / len(results)
    if avg_score >= 90: avg_grade = 'A+'
    elif avg_score >= 80: avg_grade = 'A'
    elif avg_score >= 70: avg_grade = 'B'
    else: avg_grade = 'C'
    
    print("-"*70)
    print(f"{'Average':<12} {'':<10} {avg_score:<10.1f} {avg_grade:<8}")
    print("\n" + "="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
