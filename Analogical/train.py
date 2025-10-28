from ultralytics import SPARYOLO
if __name__ == '__main__':
    model = SPARYOLO("C:/Users/19595/Desktop/Analogical-Reasoning-main/Analogical/ultralytics/cfg/models/v8/yolov8x-spar.yaml")

    model.load('yolov8x.pt')
    train_results = model.train(
        data="VisDrone_Dataset_COCO_Format/VisDrone.yaml",
        epochs=10,        # 快速测试
        imgsz=384,        # 小尺寸
        fraction=0.05,    # 5%数据
        device=[0],
        batch=1,         
        lr0=0.01,          # 新增：初始学习率
        weight_decay=0.0005,  # 新增：权重衰减
        warmup_epochs=3    # 新增：预热轮数
    )
