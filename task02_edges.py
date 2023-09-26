"""
@Author: yanzx
@Date: 2023/9/26 19:48
@Description: 边缘检测器
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image


def sobel(input_img):
    """
    索贝尔算子
    :param img:
    :return:
    """
    x = cv2.Sobel(input_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(input_img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 窗口显示
    # cv2.imshow("absX", absX)
    # cv2.imshow("absY", absY)
    # cv2.imshow("Result", dst)
    # cv2.waitKey(0)

    return absX, absY, dst


def canny_edge(input_img):
    """
    canny边缘检测
    :param input_img:
    :return:
    """
    img = cv2.GaussianBlur(input_img, (3, 3), 0)  # 用高斯平滑处理原图像降噪。
    canny = cv2.Canny(img, 50, 150)  # 最大最小阈值
    canny = cv2.convertScaleAbs(canny)
    return canny


def laplacian_edge(input_img):
    """
    拉普拉斯
    :param input_img:
    :return:
    """
    gray_lap = cv2.Laplacian(input_img, cv2.CV_16S, ksize=3)
    laplacian_dst = cv2.convertScaleAbs(gray_lap)
    for i in laplacian_dst:
        for j in i:
            print(j)
    return gray_lap


def main():
    # 读取图像
    input_img = cv2.imread('imgs/shoe.png')

    # 1. 索贝尔算子
    absX, absY, dst = sobel(input_img)
    cv2.imwrite('imgs/edges/sobel_absX_image.jpg', absX)
    cv2.imwrite('imgs/edges/sobel_absy_image.jpg', absY)
    cv2.imwrite('imgs/edges/sobel_dst_image.jpg', dst)

    # 2. canny算子
    canny_img = canny_edge(input_img)
    cv2.imwrite('imgs/edges/canny_image.jpg', canny_img)

    # 3. laplacian 算子
    lap_img = laplacian_edge(input_img)
    cv2.imwrite('imgs/edges/lap_image.jpg', lap_img)


if __name__ == '__main__':
    main()
