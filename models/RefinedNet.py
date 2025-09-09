import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class AdvancedSGM(nn.Module):
    def __init__(self, in_channels_img, num_sem_features=6):
        super(AdvancedSGM, self).__init__()
        self.num_sem_features = num_sem_features
        # 确保图像通道可以被语义特征数量整除，如果不能，则适当调整
        self.avg_channels_per_feature = in_channels_img // num_sem_features

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels_img, in_channels_img, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels_img, in_channels_img, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_feat, sem_feat):
        # 分割图像特征
        split_img_features = torch.split(img_feat, self.avg_channels_per_feature, dim=1)

        # 如果不能平均分配，最后一份可能需要合并
        if len(split_img_features) > self.num_sem_features:
            split_img_features = list(split_img_features)
            last = split_img_features.pop(-1)
            split_img_features[-1] = torch.cat((split_img_features[-1], last), dim=1)

        # 语义加权
        weighted_features = []
        for i, feature in enumerate(split_img_features):
            sem_weight = sem_feat[:, i:i + 1, :, :]
            weighted_features.append(feature * sem_weight)

        # Concat拼接特征
        concat_features = torch.cat(weighted_features, dim=1)

        # 经过两个卷积层
        output = self.conv1(concat_features)
        output = self.relu(output)
        output = self.conv2(output)

        return output


class Route2(nn.Module):
    def __init__(self, sem_channels=6, num_blocks=6):
        super(Route2, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(sem_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, sem_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


class Route1(nn.Module):
    def __init__(self, img_channels=1, sem_channels=6, num_stages=6):
        super(Route1, self).__init__()
        # 初始卷积层
        self.initial_sgm = AdvancedSGM(64, sem_channels)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 定义多阶段的卷积和 SGM 模块
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            stage = nn.ModuleDict({
                'conv': nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                'sgm': AdvancedSGM(64, sem_channels)
            })
            self.stages.append(stage)
        # 最后的卷积层
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, img_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, sem_input, sem_feats):
        out = self.initial_conv(x)
        # 初始的 SGM 融合
        # sgm = AdvancedSGM(64, sem_input.size(1))
        out = self.initial_sgm(out, sem_input)
        # 多阶段处理
        for idx, stage in enumerate(self.stages):
            residual = out
            out = stage['conv'](out)
            out = stage['sgm'](out, sem_feats[idx])
            out = out + residual  # 残差连接
        # 最终输出
        output = self.final_conv(out)
        return output


class RefinedNet(nn.Module):
    def __init__(self, img_channels=1, sem_channels=6):
        super(RefinedNet, self).__init__()
        self.route2 = Route2(sem_channels)
        self.route1 = Route1(img_channels, sem_channels)

    def forward(self, img_input, sem_input):
        sem_feats = self.route2(sem_input)
        output = self.route1(img_input, sem_input, sem_feats)
        return  output 