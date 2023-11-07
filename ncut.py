from skimage import data, segmentation, color
from skimage import graph
from matplotlib import pyplot as plt
from skimage import io
import matplotlib

matplotlib.use('TkAgg')


def ncut(img):
    """
    调用ncut包得到分割后的图片
    :param img:
    :return:
    """
    # 得到一个(m, n)的数组，其中代表每个像素点的分类，
    labels1 = segmentation.slic(img, compactness=30, n_segments=100, start_label=1)
    # 将得到的color分类进行渲染
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    return out2


def main():
    imgs_out = []
    n = 16
    for i in range(n):
        img = io.imread(f'imgs/task04/car_raw2/car%02d.png' % (i + 1))
        img = img[:, :, :3]
        out2 = ncut(img)
        imgs_out.append(out2)
    fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(6, 8))

    for i in range(4):
        for j in range(4):
            axs[i][j].imshow(imgs_out.pop())
            axs[i, j].axis('off')  # 关闭坐标轴
    plt.show()


if __name__ == '__main__':
    main()
