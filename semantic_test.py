import torch
import json
from PIL import Image
from torchvision import transforms
import models  # 假设你有模型定义
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch.nn.functional as F

def main():
    config_path = 'config.json'
    model_path = 'checkpoints/DeepLab.pth'
    image_path = 'val/SSAR/S1.png'
    config = json.load(open(config_path))
    num_classes = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, config, num_classes, device)

    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_image(image)

    semantic_tensor = get_semantic_tensor(model, input_tensor, device)
#     print("语义张量的形状:", semantic_tensor.size())

def preprocess_image(image):
    to_tensor = transforms.ToTensor()
    return to_tensor(image).unsqueeze(0)  

def load_model(model_path, config, num_classes, device):
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    if 'module' in list(checkpoint.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def get_semantic_tensor(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        # 应用 softmax 并选择具有最大概率的类别
        # semantic_tensor = F.softmax(output, dim=1).argmax(dim=1)
    return output 

if __name__=='__main__':
    main()