import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import lpips

# ==========================================
# 1. Setup Models and Transforms
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize LPIPS model for Content Preservation
# (Uses AlexNet backbone by default, standard for LPIPS)
loss_fn_vgg = lpips.LPIPS(net='alex').to(device)

# Initialize VGG19 for Style Fidelity (Gram Matrix comparison)
# We use the standard torchvision VGG19 features
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# We will extract features from these specific VGG layers for style calculation 
# (Standard Gatys et al. layers: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)
style_layers = {'0': 'relu1_1', '5': 'relu2_1', '10': 'relu3_1', '19': 'relu4_1', '28': 'relu5_1'}

def image_to_tensor(image_path, size=(512, 512)):
    """Loads an image, resizes it, and converts to a PyTorch tensor."""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # Normalize for VGG network
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    return transform(img).unsqueeze(0).to(device)

def lpips_image_to_tensor(image_path, size=(512, 512)):
    """LPIPS expects images in range [-1, 1] without VGG normalization."""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1, 1]
    ])
    return transform(img).unsqueeze(0).to(device)

# ==========================================
# 2. Metric Calculation Functions
# ==========================================

def get_features(image_tensor, model, layers):
    """Run an image forward through VGG and grab features at specific layers."""
    features = {}
    x = image_tensor
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """Calculate the Gram Matrix of a given tensor."""
    # b=batch size, d=depth(channels), h=height, w=width
    b, d, h, w = tensor.size() 
    # Reshape the tensor so we multiply features
    tensor = tensor.view(b * d, h * w) 
    # Gram matrix = tensor multiplied by its transpose
    gram = torch.mm(tensor, tensor.t()) 
    return gram

def calculate_style_loss(generated_img_path, style_img_path):
    """Calculates Style Fidelity (Lower is better)."""
    gen_tensor = image_to_tensor(generated_img_path)
    style_tensor = image_to_tensor(style_img_path)

    gen_features = get_features(gen_tensor, vgg, style_layers)
    style_features = get_features(style_tensor, vgg, style_layers)

    style_loss = 0
    # Calculate difference in Gram Matrices for every layer
    for layer in style_features:
        gen_feature = gen_features[layer]
        style_feature = style_features[layer]

        # Get Gram matrices
        gen_gram = gram_matrix(gen_feature)
        style_gram = gram_matrix(style_feature)

        # Get dimensions for normalization
        b, c, h, w = gen_feature.shape
        
        # Calculate MSE loss and add to total
        layer_style_loss = torch.mean((gen_gram - style_gram)**2)
        style_loss += layer_style_loss / (c * h * w) # Normalize by size

    return style_loss.item()

def calculate_lpips(generated_img_path, content_img_path):
    """Calculates Content Preservation using LPIPS (Lower distance = higher similarity)."""
    gen_tensor = lpips_image_to_tensor(generated_img_path)
    content_tensor = lpips_image_to_tensor(content_img_path)
    
    # Calculate distance
    with torch.no_grad():
        distance = loss_fn_vgg(gen_tensor, content_tensor)
        
    return distance.item()

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    # --- Configure your image paths here ---
    CONTENT_IMG = "input/bird.jpg"
    STYLE_FG_IMG = "style/trial.jpg"
    
    # Path to the image you just generated!
    GENERATED_IMG = "output/bird_spatial_trial_the_resevoir_at_poitiers_new3.jpg" 
    
    print(f"Evaluating: {GENERATED_IMG}")
    print("-" * 40)
    
    # 1. Calculate Content Preservation (LPIPS)
    # Compares Generated Image vs. Original Content Image
    lpips_score = calculate_lpips(GENERATED_IMG, CONTENT_IMG)
    print(f"Content Preservation (LPIPS): {lpips_score:.4f}")
    print("  *(Lower distance = Content is better preserved)*")
    print("-" * 40)
    
    # 2. Calculate Style Fidelity (Gram Matrix Loss)
    # Compares Generated Image vs. Style Target Image
    style_score = calculate_style_loss(GENERATED_IMG, STYLE_FG_IMG)
    print(f"Style Fidelity (Gram Loss):   {style_score:.4f}")
    print("  *(Lower distance = Style matches more closely)*")
    print("-" * 40)