#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 10:09:08 2025

@author: andreyvlasenko
"""


import os
import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np




class PCDDataset(Dataset):
    def __init__(self, base_dir, split, num_points=1024):
        self.num_points = num_points
        self.data = []
        self.labels = []
        self.class_map = {'Negative':0, 'Positive':1}
        for class_name in ['Negative', 'Positive']:
            folder = os.path.join(base_dir, split, class_name)
            files = [f for f in os.listdir(folder) if f.lower().endswith('.pcd')]
            for file in files:
                self.data.append(os.path.join(folder, file))
                self.labels.append(self.class_map[class_name])
        self.length = len(self.data)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.data[idx])
        pts = np.asarray(pcd.points)  # (N, 3)
        if len(pts) < self.num_points:
            # pad with zeros if not enough points
            pad = np.zeros((self.num_points - len(pts), 3))
            pts = np.vstack([pts, pad])
        else:
            # randomly sample points
            idxs = np.random.choice(len(pts), self.num_points, replace=False)
            pts = pts[idxs]
        # (num_points, 3) -> (3, num_points)
        pts = pts.T
        return torch.tensor(pts, dtype=torch.float32), self.labels[idx]
