import argparse
import torch
from torch.utils.data import DataLoader
from datasets.flying_chairs import FlyingChairs
from models.pwcnet import PWCNet
from utils.loss import MultiscaleEPELoss
from tqdm import tqdm

"""
简易验证脚本：加载 checkpoint，计算多尺度损失或 EPE（最后尺度）
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_list', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    val_set = FlyingChairs(args.val_list)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = PWCNet()
    ck = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model = model.to(device)
    model.eval()
    criterion = MultiscaleEPELoss()

    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Eval'):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            flow = batch['flow'].to(device)
            preds = model(img1, img2)
            loss = criterion(preds, flow)
            total_loss += loss.item()
    print('Validation loss:', total_loss / len(val_loader))

if __name__ == '__main__':
    main()
