import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models import RefinedNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义相关参数
BATCH_SIZE = 1
IMAGE_SIZE = 128

# 定义测试数据集
class MyTestDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.root_dir = root_dir
        self.noisy_images = os.listdir(os.path.join(root_dir, "noisy"))

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_img_name = os.path.join(self.root_dir, "noisy", self.noisy_images[idx])
        noisy_image = Image.open(noisy_img_name)
        noisy_image = self.transform(noisy_image)
        return noisy_image

# 加载测试数据集
test_path = './test'  # 假设你的测试集在这个路径
test_dataset = MyTestDataset(root_dir=test_path)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 加载训练好的模型
refinednet = RefinedNet.RefinedNet().to(device)
refinednet=(torch.load('checkpoints/RefinedNet.pth'))
refinednet.eval()

#######
sem_input = torch.randn(1, 6, 256, 256)
sem_input = sem_input.to(device)
#######

# 定义保存输出图像的函数
def save_image(tensor, path):
    image = tensor.clone().detach().cpu()
    image = torchvision.transforms.ToPILImage()(image)
    image.save(path)

# 开始测试
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with torch.no_grad():
    for idx, noisy_image in enumerate(test_loader):
        noisy_image = noisy_image.to(device)
        
        output = refinednet(noisy_image, sem_input)
        
        save_image(output.squeeze(0), os.path.join(output_dir, f'denoised_{idx}.png'))

        print(f"Saved denoised image {idx}.")

print("Testing completed.")
