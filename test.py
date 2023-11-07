import cv2

# 读取带有四通道的图片
rgba_image = cv2.imread('car.png', cv2.IMREAD_UNCHANGED)
print(rgba_image)

# 提取RGB通道
rgb_image = rgba_image[:, :, :3]

print(rgb_image)

# 保存为三通道的图片
cv2.imwrite('output_image.png', rgb_image)
