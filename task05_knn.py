"""
@Time : 2023/11/14 21:46
@Author : yanzx
@Description : 图像识别
"""

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
            img = cv2.resize(img, (255, 255))  # 调整图像大小
            # plt.imshow(img)
            # plt.show()
            images.append(img.flatten())  # 将图像展平为一维数组
            # print(img.flatten().shape)
            labels.append(class_name)
    return np.array(images), np.array(labels)


def main():
    folder_path = "imgs/scene_categories"
    X, y = load_images(folder_path)
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for k in range(3, 100):
        # 初始化KNN分类器
        knn_classifier = KNeighborsClassifier(n_neighbors=k)

        # 训练模型
        knn_classifier.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = knn_classifier.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    main()
