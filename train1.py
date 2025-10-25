from ultralytics import SPARYOLO
if __name__ == '__main__':
    model = SPARYOLO("C:/Users/19595/Desktop/Analogical-Reasoning-main/Analogical/ultralytics/cfg/models/v8/yolov8x-spar.yaml")

    model.load('yolov8x.pt')
    train_results = model.train(
        data="C:/Users/19595/Desktop/Analogical-Reasoning-main/VisDrone_Dataset_COCO_Format/VisDrone.yaml",
        epochs=80,  # number of training epochs
        imgsz=1536,  # training image size
        resume=True,
        # device=[7],
        # batch=2
        device=[0],
        batch=1
    )
