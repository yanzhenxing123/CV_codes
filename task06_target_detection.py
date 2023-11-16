import cv2

# 打开视频文件
video_path = f'imgs/videos/demo05.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频信息
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 创建背景差模型
fgbg = cv2.createBackgroundSubtractorMOG2()

# 创建视频写入对象
output_path = 'imgs/videos/processed_video/output05.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置为MP4编码
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    # 读取当前帧
    ret, frame = cap.read()
    if not ret:
        break
    # 应用背景差模型
    fgmask = fgbg.apply(frame)
    # 进行轮廓检测
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 画出目标Bounding Box
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # 过滤掉小目标
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 将原始帧写入输出视频
    out.write(frame)

# 释放视频文件，关闭窗口
cap.release()
out.release()
