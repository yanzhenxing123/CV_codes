"""
@Time : 2023/10/26 10:33
@Author : yanzx
@Description : 分割图像
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float
from loguru import logger

from sklearn.cluster import SpectralClustering

import matplotlib

matplotlib.use('TkAgg')


def kmeans(features, k, num_iters=100):
    """
    K-聚类算法实现，欧式距离
    :param features: 图片向量
    :param k: 聚类个数
    :param num_iters: 迭代次数
    :return:
    """
    N, D = features.shape
    # 随机初始化k个中心
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    for n in range(num_iters):
        f_tmp = np.tile(features, (k, 1))
        c_tmp = np.repeat(centers, N, axis=0)
        assignments = np.argmin(np.sum((f_tmp - c_tmp) ** 2, axis=1).reshape(k, N), axis=0)
        tmp = centers.copy()
        for j in range(k):
            centers[j] = np.mean(features[assignments == j], axis=0)
        if np.allclose(tmp, centers):
            break
    return assignments


def kmeans2(features, k, num_iters=100):
    """
    K-聚类算法实现，曼哈顿距离
    :param features: 图片向量
    :param k: 聚类个数
    :param num_iters: 迭代次数
    :return:
    """
    N, D = features.shape
    print(N, D)
    # 随机初始化k个中心
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    print(centers.shape)
    assignments = np.zeros(N)
    for n in range(num_iters):
        f_tmp = np.tile(features, (k, 1))
        c_tmp = np.repeat(centers, N, axis=0)
        assignments = np.argmin(np.sum(np.abs(f_tmp - c_tmp), axis=1).reshape(k, N), axis=0)
        # assignments = np.argmin(np.sum((f_tmp - c_tmp) ** 2, axis=1).reshape(k, N), axis=0)
        tmp = centers.copy()
        for j in range(k):
            centers[j] = np.mean(features[assignments == j], axis=0)
        if np.allclose(tmp, centers):
            break
    return assignments


def kmeans_cosine(features, k, num_iters=100):
    N, D = features.shape
    # 随机初始化k个中心
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    for n in range(num_iters):
        # 标准化数据
        normalized_features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
        normalized_centers = centers / np.linalg.norm(centers, axis=1)[:, np.newaxis]
        # 计算余弦相似度
        similarities = np.dot(normalized_features, normalized_centers.T)
        # 分配数据点到最相似的簇
        assignments = np.argmax(similarities, axis=1)
        tmp = centers.copy()
        for j in range(k):
            # 更新簇中心为簇内数据点的均值
            centers[j] = np.mean(features[assignments == j], axis=0)
            # 标准化新的簇中心
            centers[j] /= np.linalg.norm(centers[j])

        if np.array_equal(tmp, centers):
            break
    return assignments


# Pixel-Level Features
def color_features(img):
    """
    彩色图像转换为一维特征向量，其中每个特征向量对应一个图像像素
    :param img:
    :return:
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = img.reshape(H * W, C)
    return features


def my_features(img):
    H, W, C = img.shape
    img = img_as_float(img)
    features = img.reshape(H * W, C)
    return features


# Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    accuracy = np.mean(mask_gt == mask)
    return accuracy


def evaluate_segmentation(mask_gt, segments):
    num_segments = np.max(segments) + 1
    best_accuracy = 0
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy


def main():
    # img = io.imread('car.png')
    # H, W, C = img.shape
    # features = color_features(img)
    # assignments = kmeans_cosine(features, 3)
    # segments = assignments.reshape((H, W))
    # # Display segmentation
    # plt.imshow(segments, cmap='viridis')
    # plt.axis('off')
    # plt.show()
    start_time = time.time()
    imgs_out = []
    n = 50
    for i in range(n):
        img = io.imread(f'imgs/task04/car_raw2/car%02d.png' % (i + 1))
        img = img[:, :, :3]
        H, W, C = img.shape
        features = color_features(img)
        assignments = kmeans(features, 3)
        segments = assignments.reshape((H, W))
        imgs_out.append(segments)
    fig, axs = plt.subplots(nrows=5, ncols=10, sharex=True, sharey=True, figsize=(6, 8))
    for i in range(5):
        for j in range(10):
            axs[i][j].imshow(imgs_out.pop())
            axs[i, j].axis('off')  # 关闭坐标轴

    end_time = time.time()
    logger.info(f"cost time: {end_time - start_time}s")
    plt.show()




if __name__ == '__main__':
    main()
