# 文本区域检测，将正面生产日期、酒精度和奖项区域分割出来，并返回区域的坐标
from ultralytics import YOLO
import numpy as np

def text_area(results):

    # 生产日期
    produce_date = None
    introduce = None
    reward = None

    try:
        for result in results:
            boxes = result.boxes  # Boxes 对象，用于边界框输出
            for box in boxes:
                for cls_id in box.cls:  # 获取box信息，包括类别和坐标
                    if cls_id == 0.0:  # 判断是生产日期区域
                        produce_date = box.xyxy.squeeze().tolist()  # 将检测框坐标转化为列表
                    if cls_id == 1.0:  # 判断是酒精度介绍
                        introduce = box.xyxy.squeeze().tolist()  # 将检测框坐标转化为列表
                    if cls_id == 2.0:  # 判断是奖项区域
                        reward = box.xyxy.squeeze().tolist()  # 将检测框坐标转化为列表

        # produce_date = np.array(produce_date)
        # introduce = np.array(introduce)
        # reward = np.array(reward)
        return produce_date, introduce, reward
    except Exception as e:
        print(f"An error occurred: {e}")
        return float('inf'), float('inf'), float('inf'), float('inf')


if __name__ == '__main__':
    # 加载模型权重
    model = YOLO('models/text_area-seg.onnx')  # 从YAML构建并转移权重
    device = "cuda:0"
    img = r'test_img/Image_20240521095716766.bmp'
    results = model(img, save=True, show_boxes=False, device=device)  # 返回一个结果对象列表
    produce_date, introduce, reward = text_area(results)
    print("箱喷码生产日期：", produce_date)
    print("规格：", introduce)
    print("奖项：", reward)

    information = {
        "produce_date": produce_date,
        "introduce": introduce,
        "text_area": reward
    }
    print(information)
