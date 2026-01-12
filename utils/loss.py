import torch
import torch.nn as nn
import torch.nn.functional as F

"""
多尺度 L1 损失 (endpoint error 简化)
支持传入多尺度预测 flows（从粗到细）
gt_flow: [B,2,H,W] 为最高分辨率 ground-truth 流
pred_flows: list of predicted flows 从粗到细（每个尺度与对应特征图大小一致）
scale_weights: 权重列表或 None（默认按常见配置）
"""

class MultiscaleEPELoss(nn.Module):
    def __init__(self, weights=None):
        super(MultiscaleEPELoss, self).__init__()
        # 默认权重（示例，论文或开源实现可能不同）
        if weights is None:
            self.weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.002]
        else:
            self.weights = weights
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred_flows, gt_flow):
        # pred_flows: list from coarse->fine
        total_loss = 0.0
        b, _, H, W = gt_flow.size()
        num_scales = len(pred_flows)
        for i, pred in enumerate(pred_flows):
            # 将 gt 下采样到 pred 大小
            _, _, h, w = pred.size()
            gt_down = F.interpolate(gt_flow, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.criterion(pred, gt_down)
            weight = self.weights[i] if i < len(self.weights) else 1.0
            total_loss += weight * loss
        return total_loss
