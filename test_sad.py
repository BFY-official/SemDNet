from matplotlib import pyplot as plt
import cv2
import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models import sadnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sadnet = sadnet.SADNet().to(device)
sadnet=(torch.load('checkpoints/SADNet.pth'))
sadnet.eval()

def process_image(image_path, output_path):
    sar = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),])
    sar = transform(sar).unsqueeze(0).to(device)
    with torch.no_grad():
        desp = sadnet(sar)
    desp_image = desp.squeeze().cpu().numpy()
    cv2.imwrite(output_path, desp_image * 255)
    plt.imshow(desp_image, cmap='gray')
    plt.axis('off')
    plt.show()


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, 'SADNet_' + file_name)
            process_image(input_path, output_path)
            print(f"Processed {file_name} and saved to {output_path}")


input_folder = 'val/SAR-9_L1'
output_folder = 'results/'
process_folder(input_folder, output_folder)

print("Testing completed.")
