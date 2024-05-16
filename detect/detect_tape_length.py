# 检测侧面胶带长度是否合格
from ultralytics import YOLO
from utils.box_utils import process_class, cal_distance
import cv2
from PIL import Image
from matplotlib import pyplot as plt

# 加载模型权重
model = YOLO('../models/box_side-seg.onnx')  # 从YAML构建并转移权重
device = "cuda:0"
results = model('../test_img/box_side.jpg', save=True, show_boxes=False, device=device)  # 返回一个结果对象列表


def length_det(results):
    try:
        side_edge_points = []
        tape_top_edge_points = []
        tape_bottom_edge_points = []
        for result in results:
            boxes = result.boxes  # Boxes 对象，用于边界框输出
            masks = result.masks  # Masks 对象，用于分割掩码输出
            # 封箱侧面掩码边界点
            side_edge_points = process_class(boxes, masks, 0.0)
            # 胶带顶部掩码边界点
            tape_top_edge_points = process_class(boxes, masks, 1.0)
            # 胶带底部掩码边界点
            tape_bottom_edge_points = process_class(boxes, masks, 2.0)

            img_array = result.plot(conf=True, boxes=True, kpt_line=False)  # 在输入图像上绘制检测结果，并返回带注释的图像
            img = Image.fromarray(img_array[..., ::-1])  # PIL打开
            plt.imshow(img)
            plt.show()  # 展示图片

        print('-------------------------封箱侧面掩码四角的值-------------------------')
        print(side_edge_points)
        print('-------------------------封箱侧面顶部掩码四角的值-------------------------')
        print(tape_top_edge_points)
        print('-------------------------封箱侧面底部掩码四角的值-------------------------')
        print(tape_bottom_edge_points)

        return side_edge_points, tape_top_edge_points, tape_bottom_edge_points
    except Exception as e:
        print(f"An error occurred: {e}")
        return float('inf'), float('inf'), float('inf')


if __name__ == '__main__':
    side_edge_points, tape_top_edge_points, tape_bottom_edge_points = length_det(results)

    # 计算左侧长度，包括封箱侧面和胶带左边
    llength_box_side = cal_distance(side_edge_points[0], side_edge_points[2])
    llength_tape_bottom = cal_distance(tape_bottom_edge_points[0], tape_bottom_edge_points[2])
    llength_tape_top = cal_distance(tape_top_edge_points[0], tape_top_edge_points[2])
    # 计算右侧长度，包括封箱侧面和胶带右边
    rlength_box_side = cal_distance(side_edge_points[1], side_edge_points[3])
    rlength_tape_bottom = cal_distance(tape_bottom_edge_points[1], tape_bottom_edge_points[3])
    rlength_tape_top = cal_distance(tape_top_edge_points[1], tape_top_edge_points[3])

    print("封箱侧面左侧长度：", llength_box_side)
    print("封箱侧面右侧长度：", rlength_box_side)

    print("上部左侧胶带长度：", llength_tape_top)
    print("上部右侧胶带长度：", rlength_tape_top)

    print("下部左侧胶带长度：", llength_tape_bottom)
    print("下部右侧胶带长度：", rlength_tape_bottom)

    # TODO 后续长度判断内容，根据标准进行计算


