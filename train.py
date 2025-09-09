import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models import sadnet
from tools import Mydataset, losses
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE1 = 5e-4
BATCH_SIZE = 8
NUM_EPOCHS = 30

train_clean_dir = 'train/total_gray' 
train_noisy_dir = 'train/total_L1' 

val_clean_dir = 'val/SSAR'     
val_noisy_dir = 'val/SSAR_L1'   

train_dataset = Mydataset.Mydataset(clean_dir=train_clean_dir, noisy_dir=train_noisy_dir, IMAGE_SIZE=256)
val_dataset = Mydataset.Mydataset(clean_dir=val_clean_dir, noisy_dir=val_noisy_dir, IMAGE_SIZE=256)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

sadnet = sadnet.SADNet().to(device)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    sadnet = nn.DataParallel(sadnet)

optimizer_sad = torch.optim.Adam(sadnet.parameters(), lr=LEARNING_RATE1, betas=(0.9, 0.999))
loss_ssim = losses.log_SSIM_loss().to(device)
writer_sadnet = SummaryWriter(f"logs/sadnet")
writer_gt = SummaryWriter(f"logs/gt")

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1 / mse)

best_psnr = 0
epochs_no_improve = 0
early_stop_patience = 10 
step = 0

for epoch in range(NUM_EPOCHS):
    epoch_loss_sad = 0.0
    sadnet.train()
    for batch_idx, (noisy_image, clean_image) in enumerate(train_loader):
        noisy_image = noisy_image.to(device)
        clean_image = clean_image.to(device)
        optimizer_sad.zero_grad()
        output = sadnet(noisy_image)
        loss = loss_ssim(output, clean_image)
        loss.backward()
        optimizer_sad.step()
        epoch_loss_sad += loss.item()

        if batch_idx % 100 == 0 and batch_idx > 0:
            sadnet.eval()
            with torch.no_grad():
                desp_img = sadnet(noisy_image)
                img_grid = torchvision.utils.make_grid(desp_img, normalize=True)
                img_grid_gt = torchvision.utils.make_grid(clean_image, normalize=True)
                writer_sadnet.add_image("Train/DespImg", img_grid, global_step=step)
                writer_gt.add_image("Train/GTImg", img_grid_gt, global_step=step)
                step += 1
            sadnet.train()

    epoch_loss_sad /= len(train_loader)
    print(f"Epoch[{epoch}/{NUM_EPOCHS}] Training Loss: {epoch_loss_sad:.6f}")

    sadnet.eval()
    val_psnr = 0.0
    with torch.no_grad():
        for idx, (noisy_image, clean_image) in enumerate(val_loader):
            noisy_image = noisy_image.to(device)
            clean_image = clean_image.to(device)
            output = sadnet(noisy_image)
            psnr = calculate_psnr(output, clean_image)
            val_psnr += psnr.item()

            if idx == 0:
                img_grid = torchvision.utils.make_grid(output, normalize=True)
                img_grid_gt = torchvision.utils.make_grid(clean_image, normalize=True)
                writer_sadnet.add_image("Val/DespImg", img_grid, global_step=step)
                writer_gt.add_image("Val/GTImg", img_grid_gt, global_step=step)

    val_psnr /= len(val_loader)
    print(f"Epoch[{epoch}/{NUM_EPOCHS}] Validation PSNR: {val_psnr:.4f} dB")

    writer_sadnet.add_scalar('Loss/train', epoch_loss_sad, epoch)
    writer_sadnet.add_scalar('PSNR/val', val_psnr, epoch)

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        epochs_no_improve = 0
        torch.save(sadnet, f"checkpoints/SADNet_best.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered")
            break

    torch.save(sadnet, f"checkpoints/SADNet_epoch_{epoch}.pth")

writer_sadnet.close()
writer_gt.close()
