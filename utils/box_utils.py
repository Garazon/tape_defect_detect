# 胶带缺陷检测相关工具
import math
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# 定义两点之间距离计算的函数
def cal_distance(point1, point2):
    """
    :param point1: 第一个点，包括x和y的值
    :param point2: 第二个点，包括x和y的值
    :return: 两点之间的距离，利用sqrt函数计算
    """

    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 获取距离边界框四角距离最近的点作为实际四角的点
def get_edge_points(pos_list, mask):
    """
    :param pos_list: 边界框坐标
    :param mask: 掩码坐标
    :return: edge_points: 掩码的四角坐标列表
    """

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


# 判断齐缝线是否平分胶带,计算褶皱部分和胶带部分的面积
def is_middle_line(results):
    """
    :param   results： 模型预测结果
    :return: box_edge_points, 距离封箱面的边界框四角的点距离最近的点，形成新的列表
             tape_edge_points, 距离胶带的边界框四角的点距离最近的点，形成新的列表
             area_tape, 胶带面积
             area_tape_defect, 胶带缺陷面积
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
                        # print(box_obj_pos)
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

        # 计算褶皱程度
        if area_tape_defect == 0:
            percentage = 0
        else:  # (褶皱面积 / (胶带面积 + 褶皱面积)) * 100
            percentage = round(((area_tape_defect / (area_tape + area_tape_defect)) * 100), 2)
            percentage = str(percentage) + "%"

        # 获取四角的掩码值
        box_edge_points = get_edge_points(box_pos, box_masks_lists)

        # 判断四段距离的长度差值,用掩码值计算
        distance = []  # 四段距离分别是左上、右上、左下、右下
        for p1, p2 in zip(box_edge_points, tape_edge_points):
            dist = cal_distance(p1, p2)
            distance.append(dist)

        # 计算偏离值
        deviation_degree = (abs(distance[0] - distance[2]) / (distance[0] + distance[2]) +
                            abs(distance[1] - distance[3]) / (distance[1] + distance[3])) / 2
        deviation_degree = round((deviation_degree * 100), 2)
        deviation_degree = str(deviation_degree) + "%"

        return deviation_degree, percentage
    except Exception as e:
        print(f"An error occurred: {e}")
        return float('inf'), float('inf'), float('inf')

# 判断箱喷码平分胶带
def is_middle_code(pos_top_left_corner_y, area_mask):
    """
    :param pos_top_left_corner_y: 预测框左上角的纵坐标
    :param area_mask: 掩码区域的面积
    :return: top_ratio, 上部箱喷码差和比
             bottom_ratio, 下部箱喷码差和比
    """
    try:
        # 找到上部的箱喷码的纵坐标值
        min_indexes = sorted(range(len(pos_top_left_corner_y)), key=lambda i: pos_top_left_corner_y[i])[:2]

        # 根据索引找到对应的掩码面积
        top_code_area = [area_mask[i] for i in min_indexes]
        # 将下部的箱喷码的掩码面积加入到另一个数组中
        bottom_code_area = [value for i, value in enumerate(area_mask) if i not in min_indexes]

        # 计算上部箱喷码面积差和比
        top_ratio = round(abs(top_code_area[0] - top_code_area[1]) / (top_code_area[0] + top_code_area[1]), 2)
        # 计算下部箱喷码面积差和比
        bottom_ratio = round(abs(bottom_code_area[0] - bottom_code_area[1]) / (bottom_code_area[0] + bottom_code_area[1]), 2)

        return top_ratio, bottom_ratio
    except Exception as e:
        print(f"An error occurred: {e}")
        return float('inf'), float('inf')

# 处理每个类别的掩码和目标框
def process_class(boxes, masks, class_id):
    """
    :param boxes: 预测框
    :param masks: 掩码
    :param class_id: 类别id
    :return: edge_points, 指定类别掩码角落的点
    """
    try:
        for box, mask in zip(boxes, masks):
            if box.cls == class_id:  # 如果是指定类别，获取掩码边界点
                box_obj_pos = box.xyxy.squeeze().tolist()  # 将检测框坐标转化为列表
                box_masks_pos = mask.xy[0].tolist()
                # 存储边界框位置信息，以四个点对形式存储
                box_pos_list = [[box_obj_pos[0], box_obj_pos[1]],
                                [box_obj_pos[2], box_obj_pos[1]],
                                [box_obj_pos[0], box_obj_pos[3]],
                                [box_obj_pos[2], box_obj_pos[3]]]
                # 获取指定类别的掩码角落的点
                edge_points = get_edge_points(box_pos_list, box_masks_pos)
        return edge_points
    except Exception as e:
        print(f"An error occurred: {e}")
        return float('inf')

