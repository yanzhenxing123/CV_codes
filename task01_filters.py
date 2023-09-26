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


def gaussian_filter(img, kernel=5, sigma=1.5):
    """
    高斯滤波
    :param img:
    :param kernel:
    :param sigma:
    :return:
    """
    filter = np.zeros([kernel, kernel])
    pad = kernel // 2
    for i in range(-pad, -pad + kernel):
        for j in range(-pad, -pad + kernel):
            filter[i, j] = np.exp(- (i ** 2 + j ** 2) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
    filter /= filter.sum()

    h, w, channels = img.shape
    new_img = np.zeros([h, w, channels], dtype=np.uint8)

    for c in range(channels):
        img_channel = img[:, :, c]
        img_channel = np.pad(img_channel, ((pad, pad), (pad, pad)), 'constant')
        for i in range(h):
            for j in range(w):
                new_img[i, j, c] = np.sum(img_channel[i:i + kernel, j:j + kernel] * filter)
    return new_img


def median_filter_color(img, size):
    """
    中值滤波
    :param img:
    :param size:
    :return:
    """
    pad = size // 2
    h, w, c = img.shape
    new_img = np.zeros([h, w, c], dtype=np.uint8)
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')

    for i in range(h):
        for j in range(w):
            # 获取滤波器窗口内的像素值，并分别对三个通道进行中值滤波
            window = img[i:i + size, j:j + size, :]
            for channel in range(c):
                new_img[i, j, channel] = np.median(window[:, :, channel])
    return new_img


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def gaussian(x, sigma):
    return (1 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def bilateral_filter_color(img, radius, sigmaSpace, sigmaColor):
    """
    双边滤波
    :param img:
    :param radius:
    :param sigmaSpace:
    :param sigmaColor:
    :return:
    """
    h, w, c = img.shape
    new_img = np.zeros([h, w, c], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for channel in range(c):
                value = 0
                weight = 0
                for ri in range(-radius, radius + 1):
                    for ci in range(-radius, radius + 1):
                        y = i + ri
                        x = j + ci
                        if y < 0 or x < 0 or y >= h or x >= w:
                            continue
                        gauss_space = gaussian(distance(i, j, y, x), sigmaSpace)
                        gauss_color = gaussian(abs(img[i, j, channel] - img[y, x, channel]), sigmaColor)
                        we = gauss_space * gauss_color
                        value += we * img[y, x, channel]
                        weight += we
                value /= weight
                new_img[i, j, channel] = value
    return new_img


def sobel_x_edge_detection(image):
    """
    索贝尔 x 方向
    :param image:
    :return:
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    # 使用Sobel x卷积核进行卷积操作
    gradient_x = cv2.filter2D(gray, -1, sobel_x)
    return gradient_x


def sobel_y_edge_detection(image):
    """
    索贝尔 y 方向
    :param image:
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 定义Sobel y卷积核
    sobel_y = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # 使用Sobel y卷积核进行卷积操作
    gradient_y = cv2.filter2D(gray, -1, sobel_y)

    return gradient_y


if __name__ == '__main__':
    # 读取图像文件
    image = Image.open("imgs/shoe.png")  # 替换为你的图像文件路径
    # 将图像转换为NumPy数组
    img_array = np.array(image)
    # 读取图像
    # input_image = cv2.imread('input_image.jpg')

    # 1. 均值滤波
    # blurred_image = blur(img_array, size=3)  # 假设 img_array 是你的彩色图像数组

    # 2. 高斯滤波
    # gauss_image = gaussian_filter(img_array, kernel=5, sigma=100)

    # 3. 中值滤波
    # median_img = median_filter_color(img_array, size=10)

    # 4. 双边滤波
    # # 定义滤波器参数
    # radius = 5  # 半径
    # sigmaSpace = 20  # 空间域标准差
    # sigmaColor = 20  # 色彩域标准差
    #
    # # 对彩色图像进行双边滤波
    # # bilateral_img = bilateral_filter_color(img_array, radius, sigmaSpace, sigmaColor)
    # bilateral_img = cv2.bilateralFilter(img_array, radius, sigmaSpace, sigmaColor)

    # 5. 执行Sobel X边缘检测
    edge_x = sobel_x_edge_detection(img_array)
    edge_y = sobel_y_edge_detection(img_array)

    # 保存结果
    cv2.imwrite('edge_x_image.jpg', edge_x)
    cv2.imwrite('edge_y_image.jpg', edge_y)

    # # end. 保存处理后的图像
    # output_image = Image.fromarray(bilateral_img)
    # output_image.save("imgs/bilateral_image.jpg")  # 替换为你希望保存的文件路径和文件名
