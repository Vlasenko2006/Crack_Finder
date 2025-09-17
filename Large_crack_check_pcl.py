import os
import torch
import torch.nn as nn
from torchvision import transforms
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from Simple_pcl_nn import SimplePointNet
from scipy.interpolate import griddata

# --- SETTINGS ---
pcd_path = "wall5_coldwarm.pcd"
checkpoint_path = "checkpoints/checkpoint_epoch_9.pth"
patch_size = (227.0, 227.0)  # Patch size in point cloud units (x, y)
num_points = 1024
threshold = 0.5
output_masked_img = "crack_sample_masked.jpeg"
output_masked_pcd = "crack_sample_masked.pcd"
img_resolution = (1200, 900)  # Size for 2D projection image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimplePointNet(num_classes=2).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def project_points_to_image(pts, img_w, img_h):
    """
    Projects 3D points to 2D plane for visualization,
    normalizing x and y to fit in an image of size (img_w, img_h).
    """
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    norm_x = (pts[:, 0] - x_min) / (x_max - x_min + 1e-8)
    norm_y = (pts[:, 1] - y_min) / (y_max - y_min + 1e-8)
    px = (norm_x * (img_w - 1)).astype(np.int32)
    py = (img_h - 1 - norm_y * (img_h - 1)).astype(np.int32)  # flip y for image
    return px, py, (x_min, y_min, x_max, y_max)

def patch_bounds_grid(x_min, y_min, x_max, y_max, patch_size):
    x_bins = np.arange(x_min, x_max, patch_size[0])
    y_bins = np.arange(y_min, y_max, patch_size[1])
    cells = []
    for y0 in y_bins:
        for x0 in x_bins:
            x1 = x0 + patch_size[0]
            y1 = y0 + patch_size[1]
            cells.append((x0, y0, x1, y1))
    return cells

def classify_pcl_patches(pcd_path, model, patch_size, num_points, threshold, img_resolution, output_masked_img, output_masked_pcd):
    # --- LOAD POINT CLOUD ---
    pcd = o3d.io.read_point_cloud(pcd_path)
#    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.80)
    pts = np.asarray(pcd.points)   # (N, 3)
    pts[:,2] = - pts[:,2] 
    colors = np.asarray(pcd.colors) if pcd.colors else np.zeros((pts.shape[0], 3))
    if colors.shape[0] != pts.shape[0]:
        colors = np.zeros((pts.shape[0], 3))

    img_w, img_h = img_resolution
    px, py, (x_min, y_min, x_max, y_max) = project_points_to_image(pts, img_w, img_h)
    cells = patch_bounds_grid(x_min, y_min, x_max, y_max, patch_size)
    patch_id = 1
    crack_cells = []
    patch_numbers = []
    patch_pixel_coords = []
    patch_bounds = []
    point_mask = np.zeros(pts.shape[0], dtype=bool)

    # For labeling: map from patch_id to image center (pixel coordinates)
    patch_centers_img = {}

    # Iterate over grid cells
    for (x0, y0, x1, y1) in cells:
        print(x0, y0, x1, y1)
        mask = (
            (pts[:, 0] >= x0) & (pts[:, 0] < x1) &
            (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        )
        patch_pts = pts[mask]
        if patch_pts.shape[0] == 0:
            patch_id += 1
            continue
        # Pad or sample
        if patch_pts.shape[0] < num_points:
            pad = np.zeros((num_points - patch_pts.shape[0], 3))
            patch_pts_full = np.vstack([patch_pts, pad])
        else:
            idxs = np.random.choice(patch_pts.shape[0], num_points, replace=False)
            patch_pts_full = patch_pts[idxs]
        
        # ----- FIX: Pass points, not image, to SimplePointNet -----
        # Prepare (3, num_points) tensor for SimplePointNet
        pts_tensor = torch.tensor(patch_pts_full.T, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 3, num_points]
        with torch.no_grad():
            output = model(pts_tensor)
            pred = torch.softmax(output, dim=1)
            crack_prob = pred[0, 1].item()
            print("pred = ", pred)
            is_crack = crack_prob > threshold
        # ----------------------------------------------------------

        if is_crack:
            crack_cells.append(patch_id)
            point_mask = point_mask | mask
        # For annotation and overlay
        x_norm = ((x0 + x1) / 2 - x_min) / (x_max - x_min + 1e-8)
        y_norm = ((y0 + y1) / 2 - y_min) / (y_max - y_min + 1e-8)
        cx = int(x_norm * (img_w - 1))
        cy = int(img_h - 1 - y_norm * (img_h - 1))
        patch_centers_img[patch_id] = (cx, cy)
        # Rectangle pixel bounds
        rx0 = int(((x0 - x_min) / (x_max - x_min + 1e-8)) * (img_w - 1))
        ry0 = int(img_h - 1 - ((y0 - y_min) / (y_max - y_min + 1e-8)) * (img_h - 1))
        rx1 = int(((x1 - x_min) / (x_max - x_min + 1e-8)) * (img_w - 1))
        ry1 = int(img_h - 1 - ((y1 - y_min) / (y_max - y_min + 1e-8)) * (img_h - 1))
        patch_numbers.append(patch_id)
        patch_pixel_coords.append((rx0, min(ry0, ry1), rx1, max(ry0, ry1)))
        patch_bounds.append((x0, y0, x1, y1))
        patch_id += 1

    # --- SAVE MASKED POINT CLOUD ---
    masked_pts = pts[point_mask]
    masked_colors = colors[point_mask] if colors.shape[0] == pts.shape[0] else np.zeros((masked_pts.shape[0], 3))
    masked_pcd = o3d.geometry.PointCloud()
    masked_pcd.points = o3d.utility.Vector3dVector(masked_pts)
    masked_pcd.colors = o3d.utility.Vector3dVector(masked_colors)
    o3d.io.write_point_cloud(output_masked_pcd, masked_pcd)
    print(f"Masked PCD saved as {output_masked_pcd}")

    # --- CREATE AND SAVE MASKED IMAGE VISUALIZATION ---
    img = Image.new('RGB', (img_w, img_h), (255,255,255))
    # Plot all points
    for x, y, col in zip(px, py, colors):
        img.putpixel((x, y), tuple((col * 255).astype(np.uint8)))
    img_with_grid = img.convert('RGBA')
    overlay = Image.new('RGBA', img_with_grid.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    for i, n in enumerate(patch_numbers):
        rx0, ry0, rx1, ry1 = patch_pixel_coords[i]
        draw.rectangle([rx0, ry0, rx1, ry1], outline=(255,0,0,128), width=2)
        draw.text((rx0 + 5, ry0 + 5), str(n), fill="blue", font=font)

    img_with_grid = Image.alpha_composite(img_with_grid, overlay).convert('RGBA')

    # Mask out cracked patches
    overlay_mask = Image.new('RGBA', img_with_grid.size, (0,0,0,0))
    draw_mask = ImageDraw.Draw(overlay_mask)
    for i, n in enumerate(patch_numbers):
        if n in crack_cells:
            rx0, ry0, rx1, ry1 = patch_pixel_coords[i]
            draw_mask.rectangle([rx0, ry0, rx1, ry1], fill=(255,0,0,120))
            draw_mask.text((rx0 + 10, ry0 + 10), str(n), fill="white", font=font)
    masked_img = Image.alpha_composite(img_with_grid, overlay_mask).convert('RGB')

    plt.figure(figsize=(10, 10))
    plt.imshow(masked_img)
    plt.axis('off')
    plt.title("Masked Cracked Regions (from .pcd)", fontsize=24)
    plt.tight_layout()
    plt.savefig(output_masked_img, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"Masked JPEG visualization saved as {output_masked_img}")

    # Print cracked patch centers in image coordinates
    crack_patch_centers = {n: patch_centers_img[n] for n in patch_numbers if n in crack_cells}
    if crack_patch_centers:
        print("Crack patch centers (patch_id: x_pixel, y_pixel):")
        for patch_id, (cx, cy) in crack_patch_centers.items():
            print(f"Patch {patch_id}: ({cx}, {cy})")

    return crack_patch_centers, pts

if __name__ == "__main__":
   _, pts =  classify_pcl_patches(
        pcd_path, model, patch_size, num_points, threshold,
        img_resolution, output_masked_img, output_masked_pcd
    )
   
   points = pts
   x = points[:, 0]
   y = points[:, 1]
   z = points[:, 2]

   # 1. Build grid for surface
   grid_x, grid_y = np.meshgrid(
       np.linspace(x.min(), x.max(), 1200),
       np.linspace(y.min(), y.max(), 1900)
   )

   # 2. Interpolate z values onto grid
   grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

   # 3. Plot the surface
   fig = plt.figure(figsize=(8, 6))
   ax = fig.add_subplot(111, projection='3d')
   # Mask out NaNs if interpolation didn't cover all grid
   surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', linewidth=0, antialiased=False)
   plt.show()