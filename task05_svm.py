"""
@Time : 2023/11/15 10:24
@Author : yanzx
@Description : 
"""

import numpy as np
import cv2
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from glob import glob
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def load_images(folder_path):
    images = []
    labels = []
    for class_label in glob(folder_path + '/*'):
        class_name = class_label.split("/")[-1]
        for file_path in glob(class_label + '/*.jpg'):  # 假设图片格式为jpg
            img = cv2.imread(file_path)
            img = cv2.resize(img, (32, 32))  # 调整图像大小
            # plt.imshow(img)
            # plt.show()
            images.append(img.flatten())  # 将图像展平为一维数组
            labels.append(class_name)
    return np.array(images), np.array(labels)


def main():
    folder_path = "imgs/scene_categories"
    X, y = load_images(folder_path)
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化KNN分类器
    knn_classifier = SVC(kernel='linear')
    logger.info("create done~")
    # 训练模型
    knn_classifier.fit(X_train, y_train)
    logger.info("fit done~")
    # 在测试集上进行预测
    y_pred = knn_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    main()
