import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Mydataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, IMAGE_SIZE):
        self.transform = transforms.Compose([
            transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.clean_images = sorted(os.listdir(self.clean_dir))
        self.noisy_images = sorted(os.listdir(self.noisy_dir))


    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_img_name = os.path.join(self.clean_dir, self.clean_images[idx])
        noisy_img_name = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_image = Image.open(clean_img_name).convert('L')
        noisy_image = Image.open(noisy_img_name).convert('L')
        clean_image = self.transform(clean_image)
        noisy_image = self.transform(noisy_image)
        return noisy_image, clean_image




class Jdataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, semantic_dir, IMAGE_SIZE):
        self.transform = transforms.Compose([
            transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.semantic_dir = semantic_dir
        self.clean_images = sorted(os.listdir(self.clean_dir))
        self.noisy_images = sorted(os.listdir(self.noisy_dir))
        self.semantic_maps = sorted(os.listdir(self.semantic_dir))


    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_img_name = os.path.join(self.clean_dir, self.clean_images[idx])
        noisy_img_name = os.path.join(self.noisy_dir, self.noisy_images[idx])
        semantic_map_name = os.path.join(self.semantic_dir, self.semantic_maps[idx])
        clean_image = Image.open(clean_img_name).convert('L')
        noisy_image = Image.open(noisy_img_name).convert('L')
        semantic_map = Image.open(semantic_map_name).convert('L')
        clean_image = self.transform(clean_image)
        noisy_image = self.transform(noisy_image)
        semantic_map = self.transform(semantic_map)
        return noisy_image, clean_image, semantic_map.squeeze(0).long()