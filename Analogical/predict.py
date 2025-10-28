
from ultralytics import SPARYOLO
import cv2

# 加载模型
model = SPARYOLO("C:/Users/19595/Desktop/Analogical-Reasoning-main/runs/detect/train12/weights/best.pt")  # 可换成 yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# 读取图像
image_path = "C:/Users/19595/Desktop/Analogical-Reasoning-main/VisDrone_Dataset_COCO_Format/images/0000333_02941_d_0000016.jpg"
img = cv2.imread(image_path)

# 检测图像
results = model(img)

# 显示检测结果
results[0].show()

# 或保存检测结果图像
results[0].save(filename="output1.jpg")
