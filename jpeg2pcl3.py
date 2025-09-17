import os
import cv2
import numpy as np
import open3d as o3d

def process_and_save_pcd(img_path, out_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: failed to load {img_path}")
        return False
    img_norm = img.astype(np.float32) / img.max()
    height, width = img.shape
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    xs = xs.flatten()
    ys = ys.flatten()
    zs = img_norm.flatten()
    points = np.stack((xs, ys, zs), axis=-1)
    intensities = (img_norm.flatten() * 255).astype(np.uint8)
    intensities_reshape = intensities.reshape(-1, 1)
    colorbar = cv2.applyColorMap(intensities_reshape, cv2.COLORMAP_JET)
    colors = cv2.cvtColor(colorbar, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    colors = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)
    return True

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_folder(src_folder, dst_folder):
    ensure_dir(dst_folder)
    jpgs = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(jpgs)
    for idx, fname in enumerate(jpgs, 1):
        img_path = os.path.join(src_folder, fname)
        out_path = os.path.join(dst_folder, os.path.splitext(fname)[0] + '.pcd')
        ok = process_and_save_pcd(img_path, out_path)
        print(f"{'[OK]' if ok else '[FAIL]'} ({idx}/{total}): {fname}")

if __name__ == "__main__":
    base_src = "../Cracks/data_split/train_short"
    base_dst = "train_pcl"
    for folder in ["Negative", "Positive"]:
        src_folder = os.path.join(base_src, folder)
        dst_folder = os.path.join(base_dst, folder)
        print(f"Processing {src_folder} --> {dst_folder}")
        process_folder(src_folder, dst_folder)
    print("Done!")