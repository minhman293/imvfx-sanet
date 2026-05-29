"""
segment.py  —  Generate a fg/bg mask using SAM or fallback strategies.

Strategies:
  sam        : Segment Anything Model — works on ANY subject type (default)
  horizon    : horizontal split — top=bg (sky), bottom=fg. Best for cityscapes.
  threshold  : brightness threshold — dark pixels = fg.
  rembg      : rembg u2net — best for people/portraits on plain backgrounds.

Usage examples:
  python segment.py --content input/chicago.jpg
  python segment.py --content input/chicago.jpg --strategy sam --point 0.5 0.7
  python segment.py --content input/chicago.jpg --strategy horizon --horizon_frac 0.35
  python segment.py --content input/chicago.jpg --strategy threshold --threshold 160
  python segment.py --content input/chicago.jpg --strategy rembg
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageFilter


# ── Strategy: SAM ─────────────────────────────────────────────────────────────

def strategy_sam(img, checkpoint='sam_vit_h_4b8939.pth',
                 point_rel=(0.5, 0.6), feather=10):
    """
    Use SAM with a single foreground point prompt.
    point_rel: (x, y) as fractions of image size. (0.5, 0.5) = center.
               Move this toward your subject if default doesn't work well.
               e.g. (0.5, 0.7) = center-bottom, good for buildings/people.
    """
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    # Auto-detect model type from filename
    if 'vit_h' in checkpoint:
        model_type = 'vit_h'
    elif 'vit_l' in checkpoint:
        model_type = 'vit_l'
    elif 'vit_b' in checkpoint:
        model_type = 'vit_b'
    else:
        model_type = 'vit_h'  # default

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Loading SAM ({model_type}) on {device}...")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    img_np = np.array(img.convert('RGB'))
    predictor.set_image(img_np)

    H, W = img_np.shape[:2]
    px = int(point_rel[0] * W)
    py = int(point_rel[1] * H)
    print(f"  Point prompt: ({px}, {py}) — adjust with --point if result is wrong")

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[px, py]]),
        point_labels=np.array([1]),   # 1 = foreground
        multimask_output=True,
    )

    # Pick the mask with the highest confidence score
    best = masks[np.argmax(scores)]
    mask_arr = (best * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_arr, mode='L')

    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))

    return mask_img


# ── Strategy: horizon split ────────────────────────────────────────────────────

def strategy_horizon(img, horizon_frac=0.4, feather=20):
    """
    Horizontal split: top=bg (black), bottom=fg (white).
    Best for cityscapes/landscapes where sky is the background.
    """
    W, H = img.size
    split_y = int(H * horizon_frac)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[split_y:, :] = 255
    mask_img = Image.fromarray(mask, mode='L')
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask_img


# ── Strategy: brightness threshold ────────────────────────────────────────────

def strategy_threshold(img, threshold=180, feather=10, invert=False):
    """
    Dark pixels = fg (white in mask). Good when sky is bright, subject is dark.
    Use --invert_threshold if you want bright areas as fg instead.
    """
    gray = np.array(img.convert('L'), dtype=np.float32)
    mask = (gray > threshold if invert else gray < threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode='L')
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask_img


# ── Strategy: rembg ───────────────────────────────────────────────────────────

def strategy_rembg(img, model_name='u2net', feather=10):
    """
    rembg — good for people/portraits on plain backgrounds.
    """
    from rembg import remove, new_session
    session = new_session(model_name)
    removed = remove(img.convert('RGBA'), session=session)
    alpha = removed.split()[3]
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    return alpha


# ── Main generate function ─────────────────────────────────────────────────────

def generate_mask(content_path, output_path,
                  strategy='sam',
                  # SAM options
                  checkpoint='sam_vit_h_4b8939.pth',
                  point_rel=(0.5, 0.6),
                  # horizon options
                  horizon_frac=0.4,
                  # threshold options
                  threshold=180,
                  invert_threshold=False,
                  # rembg options
                  rembg_model='u2net',
                  # shared
                  feather=10):

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    img = Image.open(content_path).convert('RGB')
    W, H = img.size
    print(f"Image: {W}x{H} | Strategy: {strategy}")

    if strategy == 'sam':
        mask = strategy_sam(img, checkpoint=checkpoint,
                            point_rel=point_rel, feather=feather)
    elif strategy == 'horizon':
        mask = strategy_horizon(img, horizon_frac=horizon_frac, feather=feather)
    elif strategy == 'threshold':
        mask = strategy_threshold(img, threshold=threshold,
                                  feather=feather, invert=invert_threshold)
    elif strategy == 'rembg':
        mask = strategy_rembg(img, model_name=rembg_model, feather=feather)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose: sam, horizon, threshold, rembg")

    mask = mask.resize((W, H), Image.LANCZOS)
    mask.save(output_path)
    coverage = np.array(mask).mean() / 255 * 100
    print(f"Mask saved : {output_path}")
    print(f"Coverage   : fg={coverage:.1f}%  bg={100-coverage:.1f}%")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate fg/bg mask for spatial style transfer')

    parser.add_argument('--content',    type=str, required=True,
                        help='Path to content image')
    parser.add_argument('--output',     type=str, default=None,
                        help='Output mask path (default: masks/<stem>_mask.png)')
    parser.add_argument('--strategy',   type=str, default='sam',
                        choices=['sam', 'horizon', 'threshold', 'rembg'],
                        help='Segmentation strategy (default: sam)')

    # SAM
    parser.add_argument('--checkpoint', type=str,
                        default='sam_vit_h_4b8939.pth',
                        help='SAM checkpoint path (default: sam_vit_h_4b8939.pth)')
    parser.add_argument('--point',      type=float, nargs=2,
                        default=[0.5, 0.6], metavar=('X', 'Y'),
                        help='SAM point prompt as fractions of image size '
                             '(default: 0.5 0.6 = center-lower). '
                             'Example: --point 0.5 0.8 for lower subject')

    # Horizon
    parser.add_argument('--horizon_frac', type=float, default=0.4,
                        help='Horizon split from top (default 0.4 = 40%%)')

    # Threshold
    parser.add_argument('--threshold',    type=int,   default=180,
                        help='Brightness cutoff 0-255 (default 180)')
    parser.add_argument('--invert_threshold', action='store_true',
                        help='Make bright pixels fg instead of dark')

    # rembg
    parser.add_argument('--rembg_model', type=str, default='u2net')

    # Shared
    parser.add_argument('--feather',    type=int, default=10,
                        help='Edge blur radius in pixels (default 10)')

    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.content))[0]
        args.output = os.path.join('masks', f'{stem}_mask.png')

    generate_mask(
        content_path=args.content,
        output_path=args.output,
        strategy=args.strategy,
        checkpoint=args.checkpoint,
        point_rel=tuple(args.point),
        horizon_frac=args.horizon_frac,
        threshold=args.threshold,
        invert_threshold=args.invert_threshold,
        rembg_model=args.rembg_model,
        feather=args.feather,
    )