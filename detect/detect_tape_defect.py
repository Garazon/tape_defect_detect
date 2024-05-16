# 检测顶、底部胶带褶皱和齐缝线是否平分胶带
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO

from utils.box_utils import is_middle_line, cal_distance
import time

# 加载一个模型
def detect():
    model = YOLO('../models/tape-defect-seg.onnx')  # 从YAML构建并转移权重

    root = tk.Tk()  # 创建Tkinter窗口
    root.withdraw()  # 隐藏Tkinter窗口
    file_types = [('JPG', '*.jpg'), ('PNG', '*.png')]  # 设置打开文件格式

    # file_path = 'test_img/test3.jpg'
    while True:
        file_path = filedialog.askopenfilename(filetypes=file_types)  # 通过文件对话框选择图像文件

        start = time.time()
        if file_path:
            # 对图像执行推理
            results = model.predict(file_path, save=False, show_boxes=False)  # 返回一个结果对象列表
            # 加载箱子检测模型并预测箱子坐标，返回边界框四角坐标和掩码相应的坐标
            box_edge_points, tape_edge_points, area_tape, area_tape_defect = is_middle_line(results)

            # 计算褶皱程度
            if area_tape_defect == 0:
                print("胶带无褶皱！")
            else:  # (褶皱面积 / (胶带面积 + 褶皱面积)) * 100
                percentage = (area_tape_defect / (area_tape + area_tape_defect)) * 100
                print("胶带褶皱程度：{:.2f}%".format(percentage))

            # 判断齐缝线是否平分胶带
            print("纸箱四角掩码坐标：", box_edge_points)  # 打印纸箱掩码上四个角落位置信息
            print("胶带四角掩码坐标：", tape_edge_points)  # 打印胶带掩码上四个角落位置信息

            # 判断四段距离的长度差值,用掩码值计算
            distance = []  # 四段距离分别是左上、右上、左下、右下
            for p1, p2 in zip(box_edge_points, tape_edge_points):
                dist = cal_distance(p1, p2)
                distance.append(dist)

            end = time.time()
            run_time = end - start
            # 打印四段距离，后面的齐缝线判断即根据这四段距离判断
            print("四段距离为：", distance)
            print("运行时间：{:.2f}秒".format(run_time))
            # TODO

            for result in results:
                img_array = result.plot(conf=True, boxes=False, kpt_line=False)  # 在输入图像上绘制检测结果，并返回带注释的图像
                img = Image.fromarray(img_array[..., ::-1])  # PIL打开
                plt.imshow(img)
                plt.show()  # 展示图片
                txt_path = 'result.txt'
                if os.path.exists(txt_path):  # 判断result.txt是否存在，存在则清空
                    with open(txt_path, 'w') as file:
                        file.truncate(0)
                result.save_txt('result.txt')  # 保存结果
                result.save('result.jpg')  # 保存带标注的图像

            print("预测完成并保存结果图片及预测结果")
        else:
            print("未选择图片文件")
            break

if __name__ == '__main__':
    detect()



