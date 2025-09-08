#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 18:22:44 2025

@author: andreyvlasenko
"""

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
model.load_state_dict(torch.load('crack_resnet18.pth', map_location=device))
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
    n_cols = width // patch_size[0]
    n_rows = height // patch_size[1]
    patches = []
    coords = []
    numbers = []
    patch_id = 1

    # For drawing grid and numbers
    img_with_grid = img.copy()
    draw = ImageDraw.Draw(img_with_grid)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Extract patches and annotate grid
    for r in range(n_rows):
        for c in range(n_cols):
            left = c * patch_size[0]
            upper = r * patch_size[1]
            right = left + patch_size[0]
            lower = upper + patch_size[1]
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)
            coords.append((left, upper, right, lower))
            numbers.append(patch_id)
            # Draw rectangle and patch number
            draw.rectangle([left, upper, right, lower], outline="red", width=2)
            draw.text((left+5, upper+5), str(patch_id), fill="yellow", font=font)
            patch_id += 1

    # Plot image with grid numbers
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
    masked_img = img.copy()
    draw_mask = ImageDraw.Draw(masked_img)
    for i, n in enumerate(numbers):
        if n in crack_patches:
            left, upper, right, lower = coords[i]
            draw_mask.rectangle([left, upper, right, lower], fill=(255, 0, 0, 128))  # semi-transparent red overlay

    # Show masked image
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_img)
    plt.axis('off')
    plt.title("Masked Cracked Regions")
    plt.show()

    return crack_patches

# Example usage:
test_images = ['wall1.jpg', 'wall2.jpg', 'wall3.jpg']
for img_path in test_images:
    print(f"Processing {img_path}")
    crack_patches = grid_image_and_classify(img_path, model)