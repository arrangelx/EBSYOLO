from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 模型配置文件
model_yaml_path = r'./mymodules/yolov11n.yaml'

#数据集配置文件
data_yaml_path = r'./datasets/data.yaml'
if __name__ == '__main__':
    model = YOLO(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=500,
                          batch=2,
                          workers=4,
                          amp=False,  # 如果出现训练损失为Nan可以关闭amp
                          project='runs',
                          name='exp',
                          )
