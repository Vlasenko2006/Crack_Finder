import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import io
import sys
import os

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
checkpoint = torch.load('checkpoints/checkpoint_epoch_3.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

patch_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

def process_frame(frame, model, patch_size=(227, 227), threshold=0.5):
    """Apply grid, classify patches, and mask cracks on a single frame (numpy array in BGR)"""

    # Check for empty frame (debug)
    if frame is None:
        print("WARNING: Received an empty frame!")
        return None

    # Convert frame to RGB PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
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

    img_with_grid = img.convert('RGBA')
    overlay = Image.new('RGBA', img_with_grid.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
    for y in ys:
        for x in xs:
            left, upper, right, lower = x, y, x + p_w, y + p_h
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)
            coords.append((left, upper, right, lower))
            numbers.append(patch_id)
            draw.rectangle([left, upper, right, lower], outline=(255,0,0,128), width=2)
            draw.text((left + 5, upper + 5), str(patch_id), fill="blue", font=font)
            patch_id += 1
    img_with_grid = Image.alpha_composite(img_with_grid, overlay).convert('RGBA')

    # Classify each patch
    crack_patches = []
    for i, patch in enumerate(patches):
        input_tensor = patch_transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.softmax(output, dim=1)
            crack_prob = pred[0, 1].item()
            if crack_prob > threshold:
                crack_patches.append(numbers[i])

    # Mask out cracked patches
    overlay_mask = Image.new('RGBA', img_with_grid.size, (0,0,0,0))
    draw_mask = ImageDraw.Draw(overlay_mask)
    for i, n in enumerate(numbers):
        if n in crack_patches:
            left, upper, right, lower = coords[i]
            draw_mask.rectangle([left, upper, right, lower], fill=(255, 0, 0, 128))
            draw_mask.text((left + 10, upper + 10), str(n), fill="white", font=font)
    # Ensure final image is RGB before converting to numpy
    final_img = Image.alpha_composite(img_with_grid, overlay_mask).convert('RGB')
    final_img_np = np.array(final_img, dtype=np.uint8)
    return final_img_np # Return RGB numpy image for saving

def label_cracks_in_video_save_frames(input_video, frames_folder, model, patch_size=(227,227), threshold=0.5, seconds=3):
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(fps * seconds)
    frame_num = 0
    while cap.isOpened() and frame_num < total_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"WARNING: Frame {frame_num+1} could not be read or is None. Skipping.")
            break
        print(f"\nProcessing frame {frame_num+1}/{total_frames}")
        processed_frame = process_frame(frame, model, patch_size, threshold)
        if processed_frame is None:
            print(f"WARNING: Processed frame {frame_num+1} is None. Skipping write.")
            continue
        # Convert RGB (PIL style) to BGR (OpenCV style) before saving as jpg
        frame_filename = os.path.join(frames_folder, f"frame_{frame_num:04d}.jpg")
        # processed_frame is RGB numpy array, save directly using PIL
        Image.fromarray(processed_frame).save(frame_filename, quality=95)
        frame_num += 1
    cap.release()
    print(f"Saved {frame_num} processed frames to folder: {frames_folder}")

def frames_to_video(frames_folder, output_video, fps=30):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    if not frame_files:
        print("No jpg frames found in folder:", frames_folder)
        return
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for fname in frame_files:
        img = cv2.imread(os.path.join(frames_folder, fname))
        if img is None:
            print(f"WARNING: Could not read {fname}")
            continue
        out.write(img)
    out.release()
    print(f"Video saved at {output_video}")

if __name__ == "__main__":
    # Step 1: Process and save frames
    frames_folder = "processed_frames"
    label_cracks_in_video_save_frames(
        input_video="video.mp4",
        frames_folder=frames_folder,
        model=model,
        patch_size=(227, 227),
        threshold=0.5,
        seconds=3
    )
    # Step 2: Combine frames into video
    frames_to_video(
        frames_folder=frames_folder,
        output_video="video_labeled.mp4",
        fps=30 # adjust if your video has different FPS
    )
    print("Video processing complete: video_labeled.mp4")