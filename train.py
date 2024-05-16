from ultralytics import YOLO
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载模型
model = YOLO('../weights/yolov8n-seg.pt')  # 从YAML构建并转移权重
device = 'cuda:0'
model.to(device)

if __name__ == '__main__':
    # 训练模型
    results = model.train(data='box_seg.yaml', epochs=10, device=device, batch=1, workers=0)

    metrics = model.val()