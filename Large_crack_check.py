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






def pixel_to_geo(crack_patch_centers, image_size, geojson_polygon):
    """
    Convert pixel coordinates to geographic coordinates (lat, lon, alt) using GeoJSON polygon corner format.
    Args:
        crack_patch_centers: dict {patch_number: (x_pixel, y_pixel)}
        image_size: (width, height)
        geojson_polygon: dict in GeoJSON Polygon format, with 5 coordinates (corners) as [lon, lat, alt]
            coordinates[0][0] = upper-left
            coordinates[0][1] = upper-right
            coordinates[0][2] = lower-right
            coordinates[0][3] = lower-left
            coordinates[0][4] = upper-left (closed polygon)
    Returns:
        dict {patch_number: (lat, lon, alt)}
    """
    width, height = image_size
    geo_centers = {}

    # Extract corners in (lon, lat, alt) format from GeoJSON
    coords = geojson_polygon['coordinates'][0]
    # geojson: [ul, ur, lr, ll, ul]
    ul = coords[0]
    ur = coords[1]
    lr = coords[2]
    ll = coords[3]
    # Each as [lon, lat, alt]

    for patch_number, (x, y) in crack_patch_centers.items():
        # Normalize pixel coordinates to range [0, 1]
        x_norm = x / width
        y_norm = y / height

        # Interpolate top and bottom edge (left to right)
        lat_top = ul[1] + x_norm * (ur[1] - ul[1])
        lon_top = ul[0] + x_norm * (ur[0] - ul[0])
        alt_top = ul[2] + x_norm * (ur[2] - ul[2])

        lat_bottom = ll[1] + x_norm * (lr[1] - ll[1])
        lon_bottom = ll[0] + x_norm * (lr[0] - ll[0])
        alt_bottom = ll[2] + x_norm * (lr[2] - ll[2])

        # Interpolate between top and bottom edges based on y
        lat = lat_top + y_norm * (lat_bottom - lat_top)
        lon = lon_top + y_norm * (lon_bottom - lon_top)
        alt = alt_top + y_norm * (alt_bottom - alt_top)

        geo_centers[patch_number] = (lat, lon, alt)

    return geo_centers






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
    plt.title("Gridded Image with Patch Numbers",  fontsize = 24)
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
           # draw_mask.text((left + 10, upper + 10), str(n), fill="white", font=font)
       
    for i, n in enumerate(numbers):
        if n in crack_patches:
            left, upper, right, lower = coords[i]
            draw_mask.text((left + 10, upper + 10), str(n), fill="white", font=font)
    
    # Composite the overlay with the original image
    masked_img = Image.alpha_composite(masked_img, overlay).convert("RGB")
    
    # Show masked image
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_img)
    plt.axis('off')
    plt.title("Masked Cracked Regions", fontsize = 24)
    plt.show()


    print(f"Classified crack patch numbers: {crack_patches}")
    crack_patch_centers = {}
    for i, n in enumerate(numbers):
        if n in crack_patches:
            left, upper, right, lower = coords[i]
            center_x = (left + right) // 2
            center_y = (upper + lower) // 2
            crack_patch_centers[n] = (center_x, center_y)
            print(f"Patch {n}: ({center_x}, {center_y})")
            
    return crack_patch_centers




test_images = ['validate/wall1.jpg', 'validate/wall2.jpg', 'validate/wall3.jpg']

geojson_polygon = {
    "type": "Polygon",
    "coordinates": [[
        [9.73114, 53.565534, 1],   # upper-left
        [9.73160, 53.566134, 1],   # upper-right
        [9.73160, 53.566134, 0],   # lower-right
        [9.73114, 53.565534, 0],   # lower-left
        [9.73114, 53.565534, 1],   # close polygon
    ]]
}
  


for img_path in test_images:
    print(f"Processing {img_path}")
    crack_patch_centers = grid_image_and_classify(img_path, model)
    
    # Get image size (width, height)
    with Image.open(img_path) as img:
        image_size = img.size
      
    coord = pixel_to_geo(crack_patch_centers, image_size, geojson_polygon)
for patch, (lat, lon, alt) in coord.items():
    print(f"Patch_id {patch}","coordinates (lat,lon,alt):",f"({lat}, {lon}, {alt})")
    
    