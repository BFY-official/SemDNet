import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import json
import models

class Semantic(nn.Module):
    def __init__(self, config_path='config.json', model_path='checkpoints/DeepLab.pth', device=None):
        super().__init__()
        self.config = json.load(open(config_path))
        self.num_classes = 6
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        model_class = getattr(models, self.config['arch']['type'])
        model = model_class(self.num_classes, **self.config['arch']['args'])
        
        checkpoint = torch.load(model_path, map_location=self.device)
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
        
        model.to(self.device)
        model.eval()  # 设置为评估模式
        return model

    def forward(self, input_tensor):
        output = self.model(input_tensor)
        output1 = F.softmax(output, dim=1).argmax(dim=1)+1
        output1 = output1.reshape(input_tensor.size(0),1,input_tensor.size(2),input_tensor.size(3)).float()
        # print("语义张量的形状:", output.size())
        return output1, output

if __name__ == '__main__':
    config_path = 'config.json'
    model_path = 'checkpoints/DeepLab.pth'
    

    segmenter = Semantic()


    input_tensor = torch.rand(8, 3, 256, 256).cuda()

    semantic_tensor = segmenter(input_tensor)
    # print("语义张量的形状:", semantic_tensor.shape)
