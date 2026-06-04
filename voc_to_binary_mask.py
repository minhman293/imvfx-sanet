# voc_to_binary_mask.py
# Run once to convert all VOC masks to binary fg/bg masks

import os
import numpy as np
from PIL import Image

VOC_MASK_DIR = "SegmentationObject"
OUTPUT_DIR   = "masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(VOC_MASK_DIR):
    if not fname.endswith(".png"):
        continue
    mask = np.array(Image.open(os.path.join(VOC_MASK_DIR, fname)))
    # 0 = background (black), 255 = boundary (white outline), rest = objects
    binary = ((mask > 0) & (mask < 255)).astype(np.uint8) * 255
    Image.fromarray(binary).save(os.path.join(OUTPUT_DIR, fname))
    print(f"Converted: {fname}")

print("Done!")