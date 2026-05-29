"""
spatial_transfer.py  —  Spatial style transfer using SANet + fg/bg mask blending.

Applies Style A to the foreground and Style B to the background,
then composites them using a soft mask.

Usage:
    python spatial_transfer.py \
        --content input/chicago.jpg \
        --style_fg style/medieval.jpg \
        --style_bg style/wave.jpg

Outputs (in output/):
    <content>_fg_<style_fg>.jpg          (full image with fg style)
    <content>_bg_<style_bg>.jpg          (full image with bg style)
    <content>_spatial_<fg>_<bg>.jpg      (final composited result)
    masks/<content>_mask.png             (the fg/bg mask)
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.utils import save_image
from os.path import basename, splitext

# ── Reuse model definitions from Eval.py ──────────────────────────────────────
# (copy the model code here so we don't depend on importing Eval.py)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    return (feat - mean.expand(size)) / std.expand(size)

decoder_net = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)),  nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)),   nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)

vgg_net = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)),   nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)),  nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)),nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)),nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)),nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)),nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)),nn.ReLU(),
)

class SANet(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = self.sm(torch.bmm(F, G))
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(content4_1, style4_1) +
            self.upsample5_1(self.sanet5_1(content5_1, style5_1))
        ))

# ── Load models ───────────────────────────────────────────────────────────────

def load_models(decoder_path, transform_path, vgg_path, device):
    decoder = decoder_net
    transform = Transform(in_planes=512)
    vgg = vgg_net

    decoder.eval()
    transform.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path, weights_only=False))
    transform.load_state_dict(torch.load(transform_path, weights_only=False))
    vgg.load_state_dict(torch.load(vgg_path, weights_only=False))

    enc_1 = nn.Sequential(*list(vgg.children())[:4])
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])

    for net in [enc_1, enc_2, enc_3, enc_4, enc_5, transform, decoder]:
        net.to(device)

    return decoder, transform, enc_1, enc_2, enc_3, enc_4, enc_5

# ── Run SANet inference ───────────────────────────────────────────────────────

def run_sanet(content_tensor, style_tensor, decoder, transform,
              enc_1, enc_2, enc_3, enc_4, enc_5, device):
    """Run one forward pass. Returns output tensor (CPU, 1×3×H×W, values ~0–1)."""
    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)

    with torch.no_grad():
        c4 = enc_4(enc_3(enc_2(enc_1(content_tensor))))
        c5 = enc_5(c4)
        s4 = enc_4(enc_3(enc_2(enc_1(style_tensor))))
        s5 = enc_5(s4)
        out = decoder(transform(c4, s4, c5, s5))
        out = out.clamp(0, 1).cpu()

    return out

# ── Mask generation ───────────────────────────────────────────────────────────

def get_mask(content_path, mask_path, feather=10):
    """Generate or load a fg mask. Returns numpy float32 array H×W×1, values 0–1."""
    if not os.path.exists(mask_path):
        print(f"Generating mask with rembg → {mask_path}")
        from rembg import remove
        img = Image.open(content_path).convert("RGBA")
        removed = remove(img)
        alpha = removed.split()[3]  # foreground=255, background=0
        if feather > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))
        os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)
        alpha.save(mask_path)
        print(f"Mask saved: {mask_path}")
    else:
        print(f"Loading existing mask: {mask_path}")
        alpha = Image.open(mask_path).convert("L")

    return alpha

# ── Blend two stylized images using mask ─────────────────────────────────────

def blend(out_fg, out_bg, mask_pil, content_size):
    """
    out_fg, out_bg: torch tensors 1×3×H×W, values 0–1
    mask_pil: PIL grayscale image (white=fg, black=bg)
    content_size: (W, H) of the original content image
    Returns blended PIL image.
    """
    W, H = content_size

    # Convert tensors to PIL
    to_pil = transforms.ToPILImage()
    img_fg = to_pil(out_fg.squeeze(0)).resize((W, H), Image.LANCZOS)
    img_bg = to_pil(out_bg.squeeze(0)).resize((W, H), Image.LANCZOS)

    # Resize mask to match
    mask = mask_pil.resize((W, H), Image.LANCZOS).convert("L")

    # Composite: fg where mask is white, bg where mask is black
    result = Image.composite(img_fg, img_bg, mask)
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spatial style transfer with SANet")
    parser.add_argument("--content",    type=str, required=True)
    parser.add_argument("--style_fg",  type=str, required=True,
                        help="Style applied to FOREGROUND")
    parser.add_argument("--style_bg",  type=str, required=True,
                        help="Style applied to BACKGROUND")
    parser.add_argument("--mask",       type=str, default=None,
                        help="Path to existing mask PNG (optional; generated if not given)")
    parser.add_argument("--feather",    type=int, default=10,
                        help="Mask edge softness in pixels")
    parser.add_argument("--output",     type=str, default="output")
    parser.add_argument("--decoder",    type=str, default="decoder_iter_500000.pth")
    parser.add_argument("--transform",  type=str, default="transformer_iter_500000.pth")
    parser.add_argument("--vgg",        type=str, default="vgg_normalised.pth")
    parser.add_argument("--size",       type=int, default=512,
                        help="Resize shorter side of content/style to this size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs("masks", exist_ok=True)

    # ── Stems for output filenames
    c_stem  = splitext(basename(args.content))[0]
    fg_stem = splitext(basename(args.style_fg))[0]
    bg_stem = splitext(basename(args.style_bg))[0]

    # ── Load models
    print("Loading models...")
    decoder, transform, enc_1, enc_2, enc_3, enc_4, enc_5 = load_models(
        args.decoder, args.transform, args.vgg, device
    )

    # ── Prepare image transform
    def img_tf(path, size):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # Resize shorter side to `size`
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return transforms.ToTensor()(img).unsqueeze(0), img.size  # tensor, (W,H)

    content_t, content_size = img_tf(args.content, args.size)
    style_fg_t, _           = img_tf(args.style_fg, args.size)
    style_bg_t, _           = img_tf(args.style_bg, args.size)

    # ── Generate / load mask
    mask_path = args.mask or os.path.join("masks", f"{c_stem}_mask.png")
    mask_pil = get_mask(args.content, mask_path, feather=args.feather)

    # ── Run style transfer for fg style
    print("Running SANet with fg style...")
    out_fg = run_sanet(content_t, style_fg_t, decoder, transform,
                       enc_1, enc_2, enc_3, enc_4, enc_5, device)

    # ── Run style transfer for bg style
    print("Running SANet with bg style...")
    out_bg = run_sanet(content_t, style_bg_t, decoder, transform,
                       enc_1, enc_2, enc_3, enc_4, enc_5, device)

    # ── Save individual full-image outputs (useful for comparison)
    save_image(out_fg, os.path.join(args.output, f"{c_stem}_fg_{fg_stem}.jpg"))
    save_image(out_bg, os.path.join(args.output, f"{c_stem}_bg_{bg_stem}.jpg"))
    print(f"Saved full fg output and full bg output.")

    # ── Blend
    print("Blending with mask...")
    result = blend(out_fg, out_bg, mask_pil, content_size)

    out_name = os.path.join(args.output, f"{c_stem}_spatial_{fg_stem}_{bg_stem}.jpg")
    result.save(out_name)
    print(f"\n✓ Done! Spatial result saved to: {out_name}")
    print(f"  Mask saved to:                  {mask_path}")

if __name__ == "__main__":
    main()