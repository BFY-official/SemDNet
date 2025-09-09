# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image
# # 示例参数
# batch_size = 2
# num_classes = 3
# height = 4
# width = 4
#
# # 模拟网络输出
# outputs = torch.randn(batch_size, num_classes, height, width)  # 预测值
# # 模拟真实标签
# targets = torch.randint(1, 3, (batch_size, height, width))  # 真实值
#
# # 定义损失函数
# criterion = nn.CrossEntropyLoss()
#
# # 计算损失
# loss = criterion(outputs, targets)
#
# print(f'损失值: {loss.item()}')
#
#
# a1 = Image.open('../outputs/wh0001.png')
# a1 = np.asarray(a1, dtype=np.float32)
# print(a1.min())
# print()


# import os
# from PIL import Image
# import numpy as np
#
# def find_images_with_zero_pixels(folder_path):
#     images_with_zero_pixels = []
#
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         # 检查文件是否为图片类型
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
#             image_path = os.path.join(folder_path, filename)
#             try:
#                 # 打开图像
#                 with Image.open(image_path) as img:
#                     # 将图像转换为灰度模式（可选）
#                     # img = img.convert('L')
#                     # 将图像转换为 NumPy 数组
#                     img_array = np.array(img)
#
#                     # 检查图像中是否存在像素值为 0 的像素
#                     if np.any(img_array == 0):
#                         images_with_zero_pixels.append(filename)
#             except Exception as e:
#                 print(f"处理图像 {filename} 时出错：{e}")
#
#     return images_with_zero_pixels
#
# # 使用示例
# if __name__ == "__main__":
#     folder_path = '../datasets/WHDLD/Labels/'
#     images_with_zero = find_images_with_zero_pixels(folder_path)
#
#     print("包含像素值为 0 的图像有：")
#     for image_name in images_with_zero:
#         print(image_name)



import numpy as np
from PIL import Image

# 定义标签到颜色的映射 (1-6 对应的 RGB 颜色)
def get_custom_palette():
    palette = [0] * 256 * 3  # 生成256个颜色槽，每个槽3个值(RGB)

    # 定义1-6的颜色映射，剩余的保持为0
    palette[1*3:1*3+3] = [255, 0, 0]      # 类别1 -> 红色
    palette[2*3:2*3+3] = [255, 255, 0]    # 类别2 -> 黄色
    palette[3*3:3*3+3] = [192, 192, 0]    # 类别3 -> 灰黄色
    palette[4*3:4*3+3] = [0, 255, 0]      # 类别4 -> 绿色
    palette[5*3:5*3+3] = [128, 128, 128]  # 类别5 -> 灰色
    palette[6*3:6*3+3] = [0, 0, 255]      # 类别6 -> 蓝色

    return palette

# 生成单通道标签图像，类别标签为 1-6 的值
def save_label_image(label, save_path):
    """
    将语义分割的标签保存为单通道 PNG 图像并应用调色板
    :param label: 2D numpy array, 包含1到6的类别标签
    :param save_path: 保存的路径
    """
    # 创建PIL单通道图像
    label_img = Image.fromarray(label.astype(np.uint8), mode='P')

    # 获取自定义调色板
    palette = get_custom_palette()

    # 应用调色板到单通道图像
    label_img.putpalette(palette)

    # 保存带调色板的图像
    label_img.save(save_path, format='PNG')

# 示例：假设我们有一个2D numpy array 的标签（1-6）
label = np.array([
    [1, 2, 3,4,4,4],
    [4, 5, 6,2,2,2],
    [1, 6, 3,5,5,5]
], dtype=np.uint8)

# 保存带调色板的 PNG 图像
save_path = "segmentation_label.png"
save_label_image(label, save_path)




