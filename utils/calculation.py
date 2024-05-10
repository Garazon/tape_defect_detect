# 检测齐缝线是否平分胶带
import math

import cv2


# 定义两点之间距离计算的函数
def cal_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 获取距离边界框四角距离最近的点作为实际四角的点
def get_edge_points(pos_list, mask):
    edge_points = []
    for point_box in pos_list:
        min_distance = float('inf')  # 初始设为无穷大
        nearest_point = None
        for point_mask in mask:
            dist = cal_distance(point_box, point_mask)
            if dist < min_distance:
                min_distance = dist
                nearest_point = point_mask
        edge_points.append(nearest_point)
    return edge_points


# 判断齐缝线是否平分胶带
def is_middle_line(results):
    """
    :param   results: results  模型预测结果
    :return: box_edge_points, 距离封箱面的边界框四角的点距离最近的点，形成新的列表
             tape_edge_points, 距离胶带的边界框四角的点距离最近的点，形成新的列表
             area_tape, 胶带面积
             area_tape_defect 胶带缺陷面积
    """

    try:
        area_tape = 0  # 设置胶带的初始面积
        area_tape_defect = 0  # 设置胶带褶皱部分的初始面积
        # 定义存储箱面目标框的列表和掩码的列表
        box_obj_lists = []
        box_masks_lists = []
        # 获取胶带边界框和掩码列表
        for result in results:
            boxes = result.boxes  # Boxes 对象，用于边界框输出
            masks = result.masks  # Masks 对象，用于分割掩码输出
            for box, mask in zip(boxes, masks):
                for cls, contour in zip(box.cls, mask.xy):  # 获取box信息，包括类别和坐标
                    class_id = cls.item()  # 获取类别索引
                    if class_id == 0.0 or class_id == 3.0:  # 根据yaml文件更换需要判断的值，如果是盒子，不区分顶面和底面，获取掩码边界点
                        box_obj_pos = box.xyxy.squeeze().tolist()  # 将检测框坐标转化为列表
                        box_masks_pos = contour.tolist()
                        print(box_obj_pos)
                        box_obj_lists.append(box_obj_pos)  # 将检测框加入列表
                        # 把两部分箱面的掩码放进列表中
                        box_masks_lists += box_masks_pos
                    if class_id == 1.0:  # 根据yaml文件更换需要判断的值，如果是胶带，则获取掩码边界点及计算面积
                        tape_obj_pos = box.xyxy.squeeze().tolist()
                        tape_masks_pos = contour.tolist()
                        # 存储边界框位置信息，以四个点对形式存储
                        box_pos_list = [[tape_obj_pos[0], tape_obj_pos[1]],
                                        [tape_obj_pos[2], tape_obj_pos[1]],
                                        [tape_obj_pos[0], tape_obj_pos[3]],
                                        [tape_obj_pos[2], tape_obj_pos[3]]]
                        # 获取距离边界框四角的点距离最近的点，形成新的列表
                        tape_edge_points = get_edge_points(box_pos_list, tape_masks_pos)
                        area_tape = area_tape + cv2.contourArea(contour)  # 计算胶带面积
                    if class_id == 2.0:  # 根据yaml文件更换需要判断的值，如果是褶皱，则计算褶皱面积
                        area_tape_defect = area_tape_defect + cv2.contourArea(contour)  # 计算褶皱面积

        # 存储箱面目标框四角坐标
        if box_obj_lists[0][1] < box_obj_lists[1][1]:  # 判断两部分箱面目标谁在上面
            box_pos = [[box_obj_lists[0][0], box_obj_lists[0][1]], [box_obj_lists[0][2], box_obj_lists[0][1]],
                       [box_obj_lists[1][0], box_obj_lists[1][3]], [box_obj_lists[1][2], box_obj_lists[1][3]]]
        else:
            box_pos = [[box_obj_lists[1][0], box_obj_lists[1][1]], [box_obj_lists[1][2], box_obj_lists[1][1]],
                       [box_obj_lists[0][0], box_obj_lists[0][3]], [box_obj_lists[0][2], box_obj_lists[0][3]]]
        # 获取四角的掩码值
        box_edge_points = get_edge_points(box_pos, box_masks_lists)

        return box_edge_points, tape_edge_points, area_tape, area_tape_defect
    except Exception as e:
        print(f"An error occurred: {e}")
        return float('inf'), float('inf'), float('inf'), float('inf')
