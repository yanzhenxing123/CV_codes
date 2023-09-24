"""
@Author: yanzx
@Date: 2023/9/23 22:15
@Description:
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# 1. 均值滤波
def blur(img, size):
    """
    均值滤波实现
    :param img:
    :param size:
    :return:
    """
    mask = np.ones((size, size)) / size / size  # 除以size保证颜色不发生变化
    pad = size // 2
    h, w, channels = img.shape  # 获取图像的高度、宽度和通道数
    new_img = np.zeros([h, w, channels])  # 创建一个与图像形状相同的新数组
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')  # 对图像进行填充
    for i in range(h):
        for j in range(w):
            for c in range(channels):
                new_img[i, j, c] = np.sum(img[i:i + size, j:j + size, c] * mask)
    return np.uint8(new_img)


if __name__ == '__main__':
    # 读取图像文件
    image = Image.open("imgs/shoe.png")  # 替换为你的图像文件路径
    # 将图像转换为NumPy数组
    img_array = np.array(image)

    # 1. 均值滤波
    blurred_image = blur(img_array, size=3)  # 假设 img_array 是你的彩色图像数组

    # end. 保存处理后的图像
    output_image = Image.fromarray(blurred_image)
    output_image.save("imgs/blurred_image.jpg")  # 替换为你希望保存的文件路径和文件名
