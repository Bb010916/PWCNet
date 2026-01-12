import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

"""
FlyingChairs 风格数据加载器示例
支持 list 文件格式：每行包含 img1_path img2_path flow_path
flow 文件格式为 .flo (Middlebury)，如果没有 .flo，可以改为 numpy 存储等
这个 loader 提供基础读入 + 随机裁剪/归一化接口（可扩展）
"""

def read_flo_file(filename):
    # 读取 .flo (Middlebury)
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise Exception('Magic number incorrect. Invalid .flo file')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        data = np.reshape(data, (h, w, 2))
        return data

class FlyingChairs(Dataset):
    def __init__(self, list_file, transforms=None):
        """
        list_file: 每行 'img1 img2 flow'
        """
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                self.samples.append(parts[:3])
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_p, img2_p, flow_p = self.samples[idx]
        img1 = Image.open(img1_p).convert('RGB')
        img2 = Image.open(img2_p).convert('RGB')
        # 读取 flow
        flow = read_flo_file(flow_p)  # H,W,2
        # 转为 numpy
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # 转为 float32 [0,1]
        img1 = img1.transpose(2,0,1).astype('float32') / 255.0
        img2 = img2.transpose(2,0,1).astype('float32') / 255.0
        flow = flow.transpose(2,0,1).astype('float32')  # 2,H,W
        sample = {
            'img1': torch.from_numpy(img1),
            'img2': torch.from_numpy(img2),
            'flow': torch.from_numpy(flow)
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
