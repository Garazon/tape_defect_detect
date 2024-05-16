# 检测箱喷码是否平分胶带，在文件夹下遍历
from ultralytics import YOLO
import cv2
from utils.box_utils import is_middle_code
import os


# 批量检测箱喷码平分胶带情况，并将大于阈值的保存在code_defect.txt中
if __name__ == '__main__':
    model = YOLO('../models/best.pt')
    input_folder = 'F:/封箱数据/data/1_箱喷码跨胶带'
    img_list = []  # 定义空列表批量存储
    file_path = "code_defect.txt"  # 保存不合格产品
    # 遍历输入文件夹中的所有文件
    if os.path.exists(file_path):  # 判断result.txt是否存在，存在则清空
        with open(file_path, 'w') as file:
            file.truncate(0)

    for file_name in os.listdir(input_folder):
        # 读取图像
        if file_name.endswith('.jpg'):
            img_path = os.path.join(input_folder, file_name)

            img_list.append(img_path)

    # 批量检测
    for img in img_list:
        pos_top_left_corner_y = []  # 存区域的左上角坐标的y值
        area_mask = []  # 存区域的掩码面积
        results = model.predict(img, save=False, show_boxes=False)
        for result in results:
            boxes = result.boxes
            masks = result.masks
            for box, mask in zip(boxes, masks):
                box_obj_pos = box.xyxy.squeeze().tolist()  # 将检测框坐标转化为列表
                pos_top_left_corner_y.append(box_obj_pos[1])  # 获取左上角的y值
                # 设置第二个数组，存放四个区域的掩码面积
                area = cv2.contourArea(mask.xy[0])
                area_mask.append(area)

        # print("区域的左上角坐标的y值:", pos_top_left_corner_y)
        # print("区域的掩码面积:", area_mask)
        top_ratio, bottom_ratio = is_middle_code(pos_top_left_corner_y, area_mask)
        print(top_ratio, bottom_ratio)

        # 判断ratio如果超过阈值，则打印该图像名称和对应的超过的比例
        if top_ratio >= 0.2 or bottom_ratio >= 0.2:
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(
                    f"result_path:{results[0].path}, top_code_ratio: {top_ratio}, bottom_code_ratio: {bottom_ratio}\n")