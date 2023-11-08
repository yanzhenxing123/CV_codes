"""
@Time : 2023/11/7 14:28
@Author : yanzx
@Description : 工具类
"""

import os
import time

from PIL import Image
import os.path
import glob


def rename_img():
    path = 'imgs/task04/car_raw'
    files = os.listdir(path)  # 文件夹里的所有文件名存成列表list
    for i, file in enumerate(files):
        # 重点在05d，这样会自动补齐5位，不足的补零
        # 为啥是0 + i，方便后面添加，把0改了就行
        NewFileName = os.path.join(path, 'car%02d' % (19 + i) + '.png')
        OldFileName = os.path.join(path, file)
        print('第%d个文件：%s' % (i + 1, NewFileName))
        os.rename(OldFileName, NewFileName)  # 改名
        time.sleep(0.1)


def convertjpg(jpgfile, outdir, width=128 * 3, height=128 * 2):
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # rename_img()
    for jpgfile in glob.glob("imgs/task04/car_raw/*.png"):
        convertjpg(jpgfile, "imgs/task04/car_raw2")
