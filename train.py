import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from models.pwcnet import PWCNet
from datasets.flying_chairs import FlyingChairs
from utils.loss import MultiscaleEPELoss

"""
训练脚本（示例）
支持多 GPU（DataParallel）
保存 checkpoint（包含 optimizer state）
"""

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Train'):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        flow = batch['flow'].to(device)
        optimizer.zero_grad()
        pred_flows = model(img1, img2)  # list 从粗->细
        loss = criterion(pred_flows, flow)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val'):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            flow = batch['flow'].to(device)
            pred_flows = model(img1, img2)
            loss = criterion(pred_flows, flow)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', required=True, help='train list file')
    parser.add_argument('--val_list', required=False, help='val list file')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrained', default=None, help='pretrained checkpoint')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_set = FlyingChairs(args.train_list)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    val_loader = None
    if args.val_list:
        val_set = FlyingChairs(args.val_list)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    model = PWCNet()
    if args.pretrained:
        ck = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(ck['model'])
    model = torch.nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = MultiscaleEPELoss()

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
        t1 = time.time()
        print(f'Epoch {epoch} Train Loss: {train_loss:.6f} Val Loss: {val_loss if val_loss else "N/A"} Time: {t1-t0:.1f}s')

        # save checkpoint
        state = {
            'epoch': epoch,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        ckpt_path = os.path.join(args.save_dir, f'pwc_epoch{epoch:04d}.pth')
        torch.save(state, ckpt_path)

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.save_dir, 'pwc_best.pth')
            torch.save(state, best_path)

if __name__ == '__main__':
    main()
