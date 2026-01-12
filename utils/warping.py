import torch
import torch.nn.functional as F

"""
简单封装：按给定 flow 使用 grid_sample 对特征/图像进行 warp
flow: [B,2,H,W] (u,v) 表示像素偏移（以像素为单位）
x: [B,C,H,W]
"""
def warp(x, flow):
    b, c, h, w = x.size()
    # 创建网格
    # 注意 align_corners True 对应 PWC-Net 原始实现（近似），但可改动
    xx = torch.linspace(-1.0, 1.0, w, device=x.device)
    yy = torch.linspace(-1.0, 1.0, h, device=x.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=2)  # H x W x 2
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # B x H x W x 2

    # 将 flow 归一化到 [-1,1]
    flow_x = flow[:, 0, :, :] / ((w - 1.0) / 2.0)
    flow_y = flow[:, 1, :, :] / ((h - 1.0) / 2.0)
    flow_norm = torch.stack((flow_x, flow_y), dim=3)
    grid_warp = grid + flow_norm
    output = F.grid_sample(x, grid_warp, mode='bilinear', padding_mode='zeros', align_corners=True)
    return output
