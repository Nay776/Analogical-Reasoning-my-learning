from ultralytics import SPARYOLO

model = SPARYOLO("/home/lihua_zhou/nianxin/Analogical/ultralytics/cfg/models/v8/yolov8x-spar.yaml")

model.load('yolov8x.pt')
train_results = model.train(
    data="/home/lihua_zhou/nianxin/dataset/visdrone/VisDrone.yaml",  # path to dataset YAML
    epochs=80,  # number of training epochs
    imgsz=1536,  # training image size
    resume=True,
    # device=[7],
    # batch=2
    device=[4,5,6,7],
    batch=8
)