# 检测箱喷码是否平分胶带，检测一张图片
from ultralytics import YOLO
import cv2
from utils.box_utils import is_middle_code


# 检测一张图片的箱喷码是否符合要求
def detect_code(img):
    model = YOLO('../models/box_code-seg.onnx')
    pos_top_left_corner_y = []  # 存区域的左上角坐标的y值
    area_mask = []  # 存区域的掩码面积
    results = model.predict(img, save=True, show_boxes=False)
    for result in results:
        boxes = result.boxes
        masks = result.masks
        for box, mask in zip(boxes, masks):
            box_obj_pos = box.xyxy.squeeze().tolist()  # 将检测框坐标转化为列表
            pos_top_left_corner_y.append(box_obj_pos[1])  # 获取左上角的y值
            # 设置第二个数组，存放四个区域的掩码面积
            area = cv2.contourArea(mask.xy[0])
            area_mask.append(area)

    print("区域的左上角坐标的y值:", pos_top_left_corner_y)
    print("区域的掩码面积:", area_mask)
    top_ratio, bottom_ratio = is_middle_code(pos_top_left_corner_y, area_mask)
    print(top_ratio, bottom_ratio)

    # TODO 其他功能带续

if __name__ == '__main__':
    img = r'../test_img/box_code.jpg'
    detect_code(img)
