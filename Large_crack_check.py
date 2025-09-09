import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
checkpoint = torch.load('checkpoints/checkpoint_epoch_3.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Transformation for patches
patch_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])




def grid_image_and_classify(img_path, model, patch_size=(227, 227), threshold=0.5):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    p_w, p_h = patch_size

    xs = list(range(0, width - p_w + 1, p_w))
    ys = list(range(0, height - p_h + 1, p_h))
    if xs[-1] + p_w < width:
        xs.append(width - p_w)
    if ys[-1] + p_h < height:
        ys.append(height - p_h)

    patches, coords, numbers = [], [], []
    patch_id = 1

    # --- СОЗДАЕМ ПРОЗРАЧНЫЙ СЛОЙ ДЛЯ КВАДРАТОВ ---
    img_with_grid = img.convert('RGBA')
    overlay = Image.new('RGBA', img_with_grid.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except:
        font = ImageFont.load_default(64)

    for y in ys:
        for x in xs:
            left, upper, right, lower = x, y, x + p_w, y + p_h
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)
            coords.append((left, upper, right, lower))
            numbers.append(patch_id)

            draw.rectangle([left, upper, right, lower], outline=(255,0,0,128), width=2)
            # draw.rectangle([left, upper, right, lower], outline=(255,0,0,128), fill=(255,0,0,40), width=2)
            draw.text((left + 5, upper + 5), str(patch_id), fill="blue", font=font)
            patch_id += 1


    img_with_grid = Image.alpha_composite(img_with_grid, overlay).convert('RGB')

    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_grid)
    plt.axis('off')
    plt.title("Gridded Image with Patch Numbers")
    plt.show()

    # Classify each patch
    crack_patches = []
    for i, patch in enumerate(patches):
        input_tensor = patch_transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.softmax(output, dim=1)
            crack_prob = pred[0, 1].item()  # Assuming index 1 is "crack"
            is_crack = crack_prob > threshold
            if is_crack:
                crack_patches.append(numbers[i])

    print(f"Classified crack patch numbers: {crack_patches}")

    # Mask out cracked patches
    masked_img = img.convert("RGBA")  # Convert to RGBA
    overlay = Image.new("RGBA", masked_img.size, (0, 0, 0, 0))  # Transparent overlay
    draw_mask = ImageDraw.Draw(overlay)
    
    for i, n in enumerate(numbers):
        if n in crack_patches:
            left, upper, right, lower = coords[i]
            draw_mask.rectangle([left, upper, right, lower], fill=(255, 0, 0, 128))  # semi-transparent red overlay
    
    # Composite the overlay with the original image
    masked_img = Image.alpha_composite(masked_img, overlay).convert("RGB")
    
    # Show masked image
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_img)
    plt.axis('off')
    plt.title("Masked Cracked Regions")
    plt.show()


# Example usage:
test_images = ['validate/wall1.jpg', 'validate/wall2.jpg', 'validate/wall3.jpg']
for img_path in test_images:
    print(f"Processing {img_path}")
    crack_patches = grid_image_and_classify(img_path, model)