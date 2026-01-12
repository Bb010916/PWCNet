import torch
import torch.nn as nn
import torch.nn.functional as F

"""
标准 PWC-Net 精简实现（用于研究/训练）
实现要点：
- feature pyramid extractor
- warping via grid_sample
- cost volume via局部相关性（search_range）
- optical flow decoder（多尺度）
- context network（可选）
注：为便于演示，做了适度简化，但保留 PWC-Net 关键流程。
"""

# ----------------------------
# 基础 conv 单元
# ----------------------------
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=True):
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True)]
    layers.append(nn.LeakyReLU(0.1, inplace=True) if relu else nn.Identity())
    return nn.Sequential(*layers)

# ----------------------------
# 特征金字塔提取器
# ----------------------------
class FeaturePyramidExtractor(nn.Module):
    def __init__(self, num_chs=None):
        super(FeaturePyramidExtractor, self).__init__()
        # 默认通道设置，和原始论文近似（逐层增加）
        if num_chs is None:
            num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.num_chs = num_chs
        self.convs = nn.ModuleList()
        for l in range(len(num_chs)-1):
            in_ch = num_chs[l]
            out_ch = num_chs[l+1]
            seq = nn.Sequential(
                conv(in_ch, out_ch, 3, 2, 1),
                conv(out_ch, out_ch, 3, 1, 1),
                conv(out_ch, out_ch, 3, 1, 1)
            )
            self.convs.append(seq)

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        # features: 从高分辨率到低分辨率（level1, level2,...）
        return features[::-1]  # 返回从最粗到最细（便于后续处理）


# ----------------------------
# cost volume（局部相关性）
# 计算两个特征图在 search_range 范围内的相关性（逐像素）
# ----------------------------
def cost_volume(fmap1, fmap2, search_range):
    """
    fmap1: [B,C,H,W]
    fmap2: [B,C,H,W]
    search_range: int, e.g., 4 -> [-4,4] x [-4,4]
    返回: cost volume [B, (2r+1)^2, H, W]
    """
    b, c, h, w = fmap1.size()
    device = fmap1.device
    r = search_range
    # pad fmap2，以便 shift 产生空白时为 0
    pad = F.pad(fmap2, (r, r, r, r))
    cost_list = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            shifted = pad[:, :, r+dy:r+dy+h, r+dx:r+dx+w]
            # 相关性计算：逐通道乘后求均值（简化的 correlation）
            cost = (fmap1 * shifted).mean(1, keepdim=True)  # [B,1,H,W]
            cost_list.append(cost)
    cost = torch.cat(cost_list, dim=1)  # [B, (2r+1)^2, H, W]
    return cost


# ----------------------------
# 光流预测解码器单元
# ----------------------------
class FlowEstimatorDense(nn.Module):
    def __init__(self, in_ch):
        super(FlowEstimatorDense, self).__init__()
        # 使用多层卷积（带残差连接风格）
        self.conv1 = conv(in_ch, 128, 3, 1, 1)
        self.conv2 = conv(128, 128, 3, 1, 1)
        self.conv3 = conv(128, 96, 3, 1, 1)
        self.conv4 = conv(96, 64, 3, 1, 1)
        self.conv5 = conv(64, 32, 3, 1, 1)
        self.predict_flow = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        flow = self.predict_flow(x)
        return x, flow


# ----------------------------
# 上下文网络（用于 refine flow）
# ----------------------------
class ContextNetwork(nn.Module):
    def __init__(self, in_ch):
        super(ContextNetwork, self).__init__()
        self.net = nn.Sequential(
            conv(in_ch, 128, 3, 1, dilation=1, padding=1),
            conv(128, 128, 3, 1, dilation=2, padding=2),
            conv(128, 128, 3, 1, dilation=4, padding=4),
            conv(128, 96, 3, 1, dilation=8, padding=8),
            conv(96, 64, 3, 1, dilation=16, padding=16),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# PWC-Net 主体
# ----------------------------
class PWCNet(nn.Module):
    def __init__(self, search_range=4, pretrained=None):
        super(PWCNet, self).__init__()
        self.search_range = search_range
        self.extractor = FeaturePyramidExtractor()
        # 每个尺度上的 flow estimator（从粗到细）
        # 特征通道配置参考 extractor 输出与 cost volume 通道
        # extractor 输出通道列表（倒序）：[196,128,96,64,32,16] 对应 level6..level1
        feat_chs = [196, 128, 96, 64, 32, 16]
        self.num_levels = len(feat_chs)
        # 层级上的 estimator modules
        self.est_blocks = nn.ModuleList()
        for l, ch in enumerate(feat_chs):
            # cost volume 通道为 (2r+1)^2
            cost_ch = (2 * search_range + 1) ** 2
            # 输入通道 = cost_ch + feat_ch + upsampled_feat + flow_up(2)
            # 我们将 upsampled_feat 取为 previous_decoder_features (32)，保守估计
            in_ch = cost_ch + ch + (32 if l > 0 else 0) + 2
            if l == 0:
                # 最粗层没有上采样特征
                in_ch = cost_ch + ch + 2
            self.est_blocks.append(FlowEstimatorDense(in_ch))

        # context network: 接在最细尺度的 decoder 输出后用于 refine
        self.context_network = ContextNetwork(in_ch=32 + 2)

        # 权重初始化
        self._init_weights()

        if pretrained:
            self.load_state_dict(torch.load(pretrained))

    def _init_weights(self):
        # 使用 kaiming 初始化 conv 层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def warp(self, x, flow):
        # 将特征 x 用 flow（尺度与 x 匹配）进行向后采样（采样点 = 网格 + flow）
        b, c, h, w = x.size()
        # 归一化流场为 [-1,1]
        # grid: (B,H,W,2) 其中最后一维为 (x,y) 方向，范围[-1,1]
        # 注意 flow 的顺序是 (u,v) 对应 (x,y)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device)
        yy = torch.linspace(-1.0, 1.0, h, device=x.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2)  # H x W x 2
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # B x H x W x 2

        # flow 需要被归一化到 [-1,1]
        flow_x = flow[:, 0, :, :] / ((w - 1.0) / 2.0)
        flow_y = flow[:, 1, :, :] / ((h - 1.0) / 2.0)
        flow_norm = torch.stack((flow_x, flow_y), dim=3)
        grid_warp = grid + flow_norm
        output = F.grid_sample(x, grid_warp, mode='bilinear', padding_mode='zeros', align_corners=True)
        return output

    def forward(self, img1, img2):
        """
        输入 image 对 (B,3,H,W)
        返回多尺度预测 flow（从粗到细）
        """
        # 提取 feature 金字塔（返回从最粗 -> 最细）
        feats1 = self.extractor(img1)
        feats2 = self.extractor(img2)
        flows = []
        up_feat = None
        up_flow = None
        # 逐层从粗到细预测
        for lvl, (f1, f2) in enumerate(zip(feats1, feats2)):
            # 如果不是最粗层，则将上层 flow 放大两倍并 warp f2
            if up_flow is not None:
                # 上采样 flow 到当前尺度
                up_flow = F.interpolate(up_flow, size=(f1.size(2), f1.size(3)), mode='bilinear', align_corners=True) * 2.0
                # 将 f2 使用 up_flow 进行 warp
                f2_warp = self.warp(f2, up_flow)
            else:
                up_flow = torch.zeros((img1.size(0), 2, f1.size(2), f1.size(3)), device=img1.device)
                f2_warp = f2

            # 计算 cost volume
            cost = cost_volume(f1, f2_warp, self.search_range)
            cost = F.leaky_relu(cost, negative_slope=0.1)

            # concat 特征并预测
            if up_feat is None:
                # 最粗层
                x = torch.cat([cost, f1, up_flow], dim=1)
            else:
                x = torch.cat([cost, f1, up_feat, up_flow], dim=1)

            decoder_feat, flow_pred = self.est_blocks[l](x)
            # refine with context network at最后一层（最细尺度）
            if lvl == len(self.est_blocks) - 1:
                flow_refine = self.context_network(torch.cat([decoder_feat, flow_pred], dim=1))
                flow_pred = flow_pred + flow_refine

            flows.append(flow_pred)
            # 为下一层准备 upsampling 特征与 flow
            up_flow = flow_pred
            up_feat = decoder_feat

        # flows: list 从最粗 -> 最细，用户通常需要最后一个为高分辨率预测
        return flows
