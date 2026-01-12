# PWC-Net (PyTorch) - 标准实现（训练 + 推理 + 可视化）

这是一个基于 PyTorch 的 PWC-Net 实现，包含完整训练/验证脚本与推理可视化示例。代码带中文注释，适合做研究与复现实验。

主要特性
- 标准 PWC-Net 结构（特征金字塔、warp、cost volume、解码器、上下文网络）
- 多尺度损失（可用 FlyingChairs / FlyingThings 数据集）
- 支持单机多 GPU（DataParallel）
- 推理脚本与光流可视化（颜色编码）
- checkpoint 保存与恢复

依赖
- Python 3.8+
- PyTorch 1.8+
- torchvision
- tqdm
- opencv-python
- numpy
安装：
```bash
pip install -r requirements.txt
```

目录结构（示例）
- models/pwcnet.py        # PWC-Net 模型实现
- datasets/flying_chairs.py  # FlyingChairs 风格数据加载器示例
- utils/flow_viz.py       # 光流颜色编码/保存
- utils/warping.py        # 光流 warping（grid_sample 封装）
- utils/loss.py           # 多尺度损失实现
- train.py                # 训练脚本
- eval.py                 # 验证/评估脚本
- inference.py            # 单对图像推理 + 可视化
- requirements.txt
- README.md

快速开始（训练）
1. 准备数据集（按 FlyingChairs 的文件组织）。示例默认数据结构：
   dataset_root/
     image_2/0001.png
     image_2/0002.png
     image_3/0001.png
     image_3/0002.png
     flow/0001.flo
   或者你可以写一个 list 文件传入 train.py
2. 训练命令示例：
```bash
python train.py --train_list data/train_pairs.txt --val_list data/val_pairs.txt --batch_size 4 --epochs 50 --save_dir checkpoints
```

推理与可视化示例
```bash
python inference.py --img1 path/to/1.png --img2 path/to/2.png --checkpoint checkpoints/latest.pth --out flow_vis.png
```
