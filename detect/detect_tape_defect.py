# 检测顶、底部胶带褶皱和齐缝线是否平分胶带
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import json

from utils.box_utils import is_middle_line

# 加载模型
def detect():
    model = YOLO('../models/tape-defect-seg.onnx')  # 从YAML构建并转移权重

    root = tk.Tk()  # 创建Tkinter窗口
    root.withdraw()  # 隐藏Tkinter窗口
    file_types = [('JPG', '*.jpg'), ('PNG', '*.png'), ('BMP', '*.bmp')]  # 设置打开文件格式

    # file_path = 'test_img/test3.jpg'
    while True:
        file_path = filedialog.askopenfilename(filetypes=file_types)  # 通过文件对话框选择图像文件

        if file_path:
            # 对图像执行推理
            results = model.predict(file_path, save=False, show_boxes=False)  # 返回一个结果对象列表
            # 获取偏离程度，胶带褶皱程度
            deviation_degree, percentage = is_middle_line(results)

            for result in results:
                result_json = result.tojson(normalize=False)  # 将结果转为json格式
                data = json.loads(result_json)  # 以字典格式加载
                json_data = {  # 指定json保存格式
                    "ocr": {

                    },
                    "defect": {
                        "display": [
                            {
                                "name": "deviation_degree",
                                "cname": "胶带偏离程度",
                                "value": deviation_degree,
                            },
                            {
                                "name": "defect_degree",
                                "cname": "胶带褶皱程度",
                                "value": percentage,
                            }
                        ],
                        "data": data
                    }
                }
                # 将数据写入 JSON 文件
                json_path = file_path.split(".")[0] + ".json"
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(json_data, json_file, indent=4, ensure_ascii=False)  # 保存为json格式并格式化处理

            print("预测完成并保存结果图片及预测结果")
        else:
            print("未选择图片文件")
            break

if __name__ == '__main__':
    detect()



