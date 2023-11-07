import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color


def normalized_cut(image):
    # 构建相似矩阵W
    W = np.zeros((image.size, image.size))
    sigma = 1.0
    for i in range(image.size):
        for j in range(i + 1, image.size):
            diff = np.linalg.norm(image[i] - image[j])
            W[i, j] = np.exp(-diff / sigma)
            W[j, i] = W[i, j]

    # 构建度矩阵D和拉普拉斯矩阵L
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # 计算归一化割矩阵N
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
    N = D_sqrt_inv.dot(L).dot(D_sqrt_inv)

    # 谱聚类
    k = 2  # 聚类数目
    eigvals, eigvecs = np.linalg.eig(N)
    eigvecs = eigvecs[:, eigvals.argsort()[:k]]  # 取前k个特征向量
    kmeans = KMeans(n_clusters=k).fit(eigvecs)

    # 根据聚类结果分割图像
    seg = np.zeros_like(image)
    seg[kmeans.labels_ == 0] = 1
    seg = seg.reshape(image.shape[:2])

    return seg


if __name__ == '__main__':
    img = io.imread('imgs/task04/train.jpg')
    gray_img = color.rgb2gray(img)
    normalized_cut(gray_img)
