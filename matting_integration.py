"""
matting_integration.py — plug HW1 KNN matting into spatial_transfer.py.

Replaces the binary-mask + Gaussian-feather flow in spatial_transfer.py's
get_mask() with a real soft alpha matte from your HW1 KNN solver.

USAGE in spatial_transfer.py:

    from matting_integration import get_alpha_matte_from_voc

    # In main(), replace:
    #     mask_pil = get_mask(args.content, mask_path, feather=args.feather)
    # with:
    mask_pil = get_alpha_matte_from_voc(
        content_path = args.content,
        voc_mask_path = args.mask,         # the binary VOC mask path
        trimap_save_path = f"masks/{c_stem}_trimap.png",
        alpha_save_path  = f"masks/{c_stem}_alpha.png",
    )

The PIL.Image.composite() call in blend() already does:
    result = alpha * img_fg + (1 - alpha) * img_bg
so swapping in a grayscale alpha matte for the feathered binary mask is enough.
"""

import os
import cv2
import numpy as np
from PIL import Image

from matting import knn_matting


# ─── Step 1: VOC binary mask → trimap ─────────────────────────────────────────
def binary_to_trimap(binary_mask, erode_radius=8, dilate_radius=18):
    """
    binary_mask : np.uint8 H×W, values {0, 255}
    returns     : np.uint8 H×W, values {0, 128, 255}
                  0   = definite background
                  128 = unknown (matting solves here)
                  255 = definite foreground
    """
    k_e = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erode_radius + 1,  2 * erode_radius + 1))
    k_d = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))

    definite_fg = cv2.erode(binary_mask,  k_e)
    possible_fg = cv2.dilate(binary_mask, k_d)

    trimap = np.full_like(binary_mask, 128, dtype=np.uint8)
    trimap[possible_fg ==   0] =   0
    trimap[definite_fg == 255] = 255
    return trimap


# ─── Step 2: VOC mask (any color encoding) → binary ──────────────────────────
def voc_mask_to_binary(voc_mask_path):
    """
    PASCAL VOC segmentation masks use indexed colors. Treat any non-black
    pixel as foreground. If you have multiple classes in one mask and only
    want one of them, pass that specific RGB color instead.
    """
    m = cv2.imread(voc_mask_path)
    if m is None:
        raise FileNotFoundError(voc_mask_path)
    if m.ndim == 3:
        binary = (m.sum(axis=2) > 30).astype(np.uint8) * 255
    else:
        binary = (m > 30).astype(np.uint8) * 255
    return binary


# ─── Step 3: orchestrate the full mask → alpha-matte pipeline ────────────────
def get_alpha_matte_from_voc(content_path,
                             voc_mask_path,
                             trimap_save_path=None,
                             alpha_save_path=None,
                             # trimap params
                             erode_radius=8,
                             dilate_radius=18,
                             # knn_matting params (tune per image)
                             K=10,
                             spatial_weight=0.05,
                             sigma=0.10,
                             my_lambda=100,
                             use_hsv=True,
                             use_edges=False,
                             use_seed_distance=False,
                             # performance: downsample for matting, then upsample
                             matting_size=None):
    """
    Returns a PIL grayscale Image (the alpha matte) sized to the content image,
    ready to drop into PIL.Image.composite(stylized_fg, stylized_bg, alpha).
    """
    # Read content + VOC mask
    img    = cv2.imread(content_path)
    if img is None:
        raise FileNotFoundError(content_path)
    binary = voc_mask_to_binary(voc_mask_path)

    H, W = img.shape[:2]
    assert binary.shape == (H, W), \
        f"mask shape {binary.shape} doesn't match content {(H, W)}"

    # Optional downsample so KNN matting runs in reasonable time on big images.
    # At 500×375 you don't need this. At 1080p you definitely do (KNN matting
    # is O(N log N) for KNN search + sparse solve).
    if matting_size is not None:
        scale = matting_size / max(H, W)
        new_w = int(W * scale)
        new_h = int(H * scale)
        img_small    = cv2.resize(img,    (new_w, new_h), interpolation=cv2.INTER_AREA)
        binary_small = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        img_small, binary_small = img, binary

    # Build trimap
    trimap = binary_to_trimap(binary_small,
                              erode_radius=erode_radius,
                              dilate_radius=dilate_radius)
    if trimap_save_path:
        os.makedirs(os.path.dirname(trimap_save_path) or ".", exist_ok=True)
        cv2.imwrite(trimap_save_path, trimap)

    # Run HW1 KNN matting. Returns float64 array H×W ∈ [0, 1].
    print(f"[matting] running HW1 KNN matting on {img_small.shape[:2]}...")
    alpha = knn_matting(
        image_bgr        = img_small,
        trimap           = trimap,
        K                = K,
        spatial_weight   = spatial_weight,
        sigma            = sigma,
        my_lambda        = my_lambda,
        use_hsv          = use_hsv,
        use_edges        = use_edges,
        use_seed_distance= use_seed_distance,
    )

    # Upsample alpha back to full resolution if we downsampled
    if matting_size is not None:
        alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 1.0)

    alpha_u8 = (alpha * 255).astype(np.uint8)
    if alpha_save_path:
        os.makedirs(os.path.dirname(alpha_save_path) or ".", exist_ok=True)
        cv2.imwrite(alpha_save_path, alpha_u8)

    return Image.fromarray(alpha_u8, mode="L")

def get_alpha_matte_from_trimap(content_path,
                                trimap_path,
                                alpha_save_path=None,
                                # knn_matting params (tune per image)
                                K=10,
                                spatial_weight=0.05,
                                sigma=0.10,
                                my_lambda=100,
                                use_hsv=True,
                                use_edges=False,
                                use_seed_distance=False,
                                matting_size=None):
    """
    Like get_alpha_matte_from_voc, but takes a pre-made trimap directly.
    Use this when you already have a hand-painted or HW1-style trimap
    with values {0, 128, 255}.
    """
    img = cv2.imread(content_path)
    if img is None:
        raise FileNotFoundError(content_path)

    trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        raise FileNotFoundError(trimap_path)

    H, W = img.shape[:2]
    assert trimap.shape == (H, W), \
        f"trimap shape {trimap.shape} doesn't match content {(H, W)}"

    # Optional downsample for big images (use INTER_NEAREST to preserve {0,128,255})
    if matting_size is not None:
        scale = matting_size / max(H, W)
        new_w, new_h = int(W * scale), int(H * scale)
        img_small    = cv2.resize(img,    (new_w, new_h), interpolation=cv2.INTER_AREA)
        trimap_small = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        img_small, trimap_small = img, trimap

    print(f"[matting] running HW1 KNN matting on {img_small.shape[:2]}...")
    alpha = knn_matting(
        image_bgr        = img_small,
        trimap           = trimap_small,
        K                = K,
        spatial_weight   = spatial_weight,
        sigma            = sigma,
        my_lambda        = my_lambda,
        use_hsv          = use_hsv,
        use_edges        = use_edges,
        use_seed_distance= use_seed_distance,
    )

    if matting_size is not None:
        alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 1.0)

    alpha_u8 = (alpha * 255).astype(np.uint8)
    if alpha_save_path:
        os.makedirs(os.path.dirname(alpha_save_path) or ".", exist_ok=True)
        cv2.imwrite(alpha_save_path, alpha_u8)

    return Image.fromarray(alpha_u8, mode="L")