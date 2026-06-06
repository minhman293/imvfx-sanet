"""
spatial_transfer.py  —  Spatial style transfer using SANet + fg/bg mask blending.
Now includes automated evaluation (LPIPS, Gram Loss) and CSV logging.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torchvision.utils import save_image
from os.path import basename, splitext
import lpips
from matting_integration import get_alpha_matte_from_voc, get_alpha_matte_from_trimap

# ==========================================
# 1. EVALUATION HELPERS
# ==========================================
def load_eval_models(device):
    """Loads standard LPIPS and VGG19 models for objective evaluation."""
    print("Loading evaluation models (LPIPS & VGG19)...")
    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)
    
    vgg_eval = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for param in vgg_eval.parameters():
        param.requires_grad = False
    return loss_fn_vgg, vgg_eval

def get_features(image_tensor, model, layers):
    features = {}
    x = image_tensor
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)
    return torch.mm(tensor, tensor.t())

def calculate_style_loss(gen_path, style_path, vgg_eval, device):
    style_layers = {'0': 'relu1_1', '5': 'relu2_1', '10': 'relu3_1', '19': 'relu4_1', '28': 'relu5_1'}
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gen_t = transform(Image.open(gen_path).convert('RGB')).unsqueeze(0).to(device)
    style_t = transform(Image.open(style_path).convert('RGB')).unsqueeze(0).to(device)

    gen_features = get_features(gen_t, vgg_eval, style_layers)
    style_features = get_features(style_t, vgg_eval, style_layers)

    style_loss = 0
    for layer in style_features:
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = gram_matrix(style_features[layer])
        b, c, h, w = gen_features[layer].shape
        style_loss += torch.mean((gen_gram - style_gram)**2) / (c * h * w)
    return style_loss.item()

def calculate_lpips(gen_path, content_path, loss_fn_vgg, device):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    gen_t = transform(Image.open(gen_path).convert('RGB')).unsqueeze(0).to(device)
    content_t = transform(Image.open(content_path).convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        dist = loss_fn_vgg(gen_t, content_t)
    return dist.item()

# ==========================================
# 2. SANET MODELS
# ==========================================

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
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

    def forward(self, content, style, mask=None): 
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c_g, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g)
        
        attn = torch.bmm(F, G) 
        
        if mask is not None:
             mask_resized = torch.nn.functional.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False) 
             mask_flat = mask_resized.view(b, 1, h * w).permute(0, 2, 1)
             attn = attn + (mask_flat - 1) * 1e9  
             
        S = self.sm(attn)
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

    def forward(self, content4_1, style4_1, content5_1, style5_1, mask=None):
        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(content4_1, style4_1, mask) +
            self.upsample5_1(self.sanet5_1(content5_1, style5_1, mask))
        ))

def load_sanet_models(decoder_path, transform_path, vgg_path, device):
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

def run_sanet(content_tensor, style_tensor, mask_tensor, decoder, transform,
              enc_1, enc_2, enc_3, enc_4, enc_5, device):
    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)
    
    if mask_tensor is not None:
        mask_tensor = mask_tensor.to(device)

    with torch.no_grad():
        c4 = enc_4(enc_3(enc_2(enc_1(content_tensor))))
        c5 = enc_5(c4)
        s4 = enc_4(enc_3(enc_2(enc_1(style_tensor))))
        s5 = enc_5(s4)
        
        out = decoder(transform(c4, s4, c5, s5, mask=mask_tensor))
        out = out.clamp(0, 1).cpu()

    return out

def blend(out_fg, out_bg, mask_pil, content_size):
    W, H = content_size
    to_pil = transforms.ToPILImage()
    img_fg = to_pil(out_fg.squeeze(0)).resize((W, H), Image.LANCZOS)
    img_bg = to_pil(out_bg.squeeze(0)).resize((W, H), Image.LANCZOS)
    mask = mask_pil.resize((W, H), Image.LANCZOS).convert("L")
    return Image.composite(img_fg, img_bg, mask)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Spatial style transfer with SANet + Evaluation")
    parser.add_argument("--content",    type=str, required=True)
    parser.add_argument("--style_fg",  type=str, required=True)
    parser.add_argument("--style_bg",  type=str, required=True)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--trimap", type=str, default=None)
    parser.add_argument("--output",     type=str, default="output")
    parser.add_argument("--decoder",    type=str, default="decoder_iter_500000.pth")
    parser.add_argument("--transform",  type=str, default="transformer_iter_500000.pth")
    parser.add_argument("--vgg",        type=str, default="vgg_normalised.pth")
    parser.add_argument("--size",       type=int, default=512)
    parser.add_argument("--csv_log",    type=str, default="evaluation_results.csv", help="CSV log filename")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs("masks", exist_ok=True)

    c_stem  = splitext(basename(args.content))[0]
    fg_stem = splitext(basename(args.style_fg))[0]
    bg_stem = splitext(basename(args.style_bg))[0]

    print("Loading SANet models...")
    decoder, transform, enc_1, enc_2, enc_3, enc_4, enc_5 = load_sanet_models(
        args.decoder, args.transform, args.vgg, device
    )
    
    # Pre-load evaluation models
    loss_fn_vgg, vgg_eval = load_eval_models(device)

    def img_tf(path, size):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return transforms.ToTensor()(img).unsqueeze(0), img.size 

    content_t, content_size = img_tf(args.content, args.size)
    style_fg_t, _           = img_tf(args.style_fg, args.size)
    style_bg_t, _           = img_tf(args.style_bg, args.size)

    trimap_path = f"masks/{c_stem}_trimap.png"
    alpha_path  = f"masks/{c_stem}_alpha.png"

    if args.trimap:
        mask_pil = get_alpha_matte_from_trimap(
            content_path=args.content, trimap_path=args.trimap, alpha_save_path=alpha_path,
            K=12, spatial_weight=0.08, sigma=0.08, my_lambda=100,
            use_hsv=False, use_edges=False, use_seed_distance=False,
        )
    elif args.mask:
        mask_pil = get_alpha_matte_from_voc(
            content_path=args.content, voc_mask_path=args.mask,
            trimap_save_path=trimap_path, alpha_save_path=alpha_path,
        )
    else:
        raise SystemExit("Pass either --mask or --trimap")

    binary_mask_pil = Image.open(args.mask).convert("L")
    binary_mask_pil = binary_mask_pil.resize((content_size[0], content_size[1]), Image.NEAREST)
    
    mask_tensor = transforms.ToTensor()(binary_mask_pil).unsqueeze(0).to(device)
    mask_tensor_fg = (mask_tensor > 0.5).float()  
    mask_tensor_bg = 1.0 - mask_tensor_fg         

    print("Running SANet with fg style...")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time_fg = time.perf_counter()

    out_fg = run_sanet(content_t, style_fg_t, mask_tensor_fg, decoder, transform,
                       enc_1, enc_2, enc_3, enc_4, enc_5, device)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    fg_runtime = time.perf_counter() - start_time_fg
    print(f"✓ Foreground processing took: {fg_runtime:.4f} seconds")

    style_strength_fg = 0.8 
    out_fg = torch.nn.functional.interpolate(out_fg, size=(content_t.shape[2], content_t.shape[3]), mode='bilinear', align_corners=False)
    out_fg = (out_fg * style_strength_fg) + (content_t.cpu() * (1.0 - style_strength_fg))

    print("Running SANet with bg style...")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time_bg = time.perf_counter()

    out_bg = run_sanet(content_t, style_bg_t, mask_tensor_bg, decoder, transform,
                       enc_1, enc_2, enc_3, enc_4, enc_5, device)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    bg_runtime = time.perf_counter() - start_time_bg
    print(f"✓ Background processing took: {bg_runtime:.4f} seconds")

    style_strength_bg = 1.0
    out_bg = torch.nn.functional.interpolate(out_bg, size=(content_t.shape[2], content_t.shape[3]), mode='bilinear', align_corners=False)
    out_bg = (out_bg * style_strength_bg) + (content_t.cpu() * (1.0 - style_strength_bg))

    print("Blending with mask...")
    result = blend(out_fg, out_bg, mask_pil, content_size)

    out_name = os.path.join(args.output, f"{c_stem}_spatial_{fg_stem}_{bg_stem}.jpg")
    result.save(out_name)
    print(f"\n✓ Spatial result saved to: {out_name}")

    # ==========================================
    # EVALUATION & LOGGING
    # ==========================================
    print("\nRunning Evaluation Metrics...")
    lpips_score = calculate_lpips(out_name, args.content, loss_fn_vgg, device)
    fg_style_loss = calculate_style_loss(out_name, args.style_fg, vgg_eval, device)
    bg_style_loss = calculate_style_loss(out_name, args.style_bg, vgg_eval, device)
    total_runtime = fg_runtime + bg_runtime

    print(f"  LPIPS: {lpips_score:.4f}")
    print(f"  FG Gram Loss: {fg_style_loss:.4f}")
    print(f"  BG Gram Loss: {bg_style_loss:.4f}")

    file_exists = os.path.isfile(args.csv_log)
    with open(args.csv_log, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Content Image", "FG Style", "BG Style", "Output File", 
                             "FG Runtime (s)", "BG Runtime (s)", "Total Runtime (s)", 
                             "LPIPS", "FG Gram Loss", "BG Gram Loss"])
        writer.writerow([basename(args.content), basename(args.style_fg), basename(args.style_bg), 
                         basename(out_name), round(fg_runtime, 4), round(bg_runtime, 4), 
                         round(total_runtime, 4), round(lpips_score, 4), 
                         round(fg_style_loss, 4), round(bg_style_loss, 4)])
    
    print(f"✓ Results appended to {args.csv_log}")

if __name__ == "__main__":
    main()