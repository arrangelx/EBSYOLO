from ultralytics import YOLO
if __name__ == '__main__':

    model = YOLO('./yolo11n.pt')
    # #model = YOLO('runs/detect/coco源配置/weights/best.pt')
    # #训练模型
    model.train(data='./datasets/data.yaml', workers=0, epochs=100, batch=4)


