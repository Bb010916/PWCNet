import argparse
import cv2
import torch
import numpy as np
from models.pwcnet import PWCNet
from utils.flow_viz import flow_to_image, write_flow_png

"""
单对图像推理并可视化（保存颜色编码图）
输入图像会被 resize 到  H,W 可被 64 整除（PWC-Net 对输入尺度敏感）
"""

def preprocess(img):
    # img: HxWx3 BGR (cv2读入)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # 将尺寸调整为能被 64 整除（方便金字塔）
    new_h = (h + 63) // 64 * 64
    new_w = (w + 63) // 64 * 64
    img_resized = cv2.resize(img, (new_w, new_h))
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = img_resized.transpose(2,0,1)[None,:,:,:]
    return img_resized, (h, w), (new_h, new_w)

def postprocess_flow(flow, orig_size, resized_size):
    # flow: [2,Hr,Wr] -> 裁切并按比例缩放到原始尺寸
    h0, w0 = orig_size
    hr, wr = resized_size
    flow = flow.transpose(1,2,0)  # Hr x Wr x 2
    # 裁切到原始比例（左上角）
    flow = flow[:h0, :w0, :]
    # 若 resized != original，则需要按比例缩放 flow 分量
    sx = float(w0) / float(wr)
    sy = float(h0) / float(hr)
    flow[:,:,0] *= (1.0 / sx)  # x 分量随着横向缩放反向调整
    flow[:,:,1] *= (1.0 / sy)
    return flow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', required=True)
    parser.add_argument('--img2', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=False, default='flow_vis.png')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    img1_p, orig_size, resized_size = preprocess(img1)
    img2_p, _, _ = preprocess(img2)
    img1_t = torch.from_numpy(img1_p).to(device)
    img2_t = torch.from_numpy(img2_p).to(device)

    model = PWCNet()
    ck = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(img1_t, img2_t)  # list coarse->fine
        flow = preds[-1][0].cpu().numpy()  # 2,Hr,Wr
        flow = flow.transpose(1,2,0)  # Hr,Wr,2
        flow = postprocess_flow(flow, orig_size, resized_size)
        write_flow_png(flow, args.out)
        print('Saved flow visualization to', args.out)

if __name__ == '__main__':
    main()
