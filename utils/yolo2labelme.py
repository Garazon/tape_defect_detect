import os
import glob
import numpy as np
import cv2
import json
# 可以将yolov8生成的txt格式的标注转为json，可以使用labelme查看标注
# 该方法可以用于辅助数据标注
def convert_txt_to_labelme_json(txt_path, image_path, output_dir, class_name, image_fmt='.jpg' ):
    txts = glob.glob(os.path.join(txt_path, "*.txt"))
    for txt in txts:
        labelme_json = {
            'version': '4.5.1',
            'flags': {},
            'shapes': [],
            'imagePath': None,
            'imageData': None,
            'imageHeight': None,
            'imageWidth': None,
        }
        txt_name = os.path.basename(txt)
        image_name = txt_name.split(".")[0] + image_fmt
        labelme_json['imagePath'] = image_name
        image_name = os.path.join(image_path, image_name)
        if not os.path.exists(image_name):
            raise Exception('txt 文件={},找不到对应的图像={}'.format(txt, image_name))
        image = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        labelme_json['imageHeight'] = h
        labelme_json['imageWidth'] = w
        with open(txt, 'r') as t:
            lines = t.readlines()
            for line in lines:
                point_list = []
                content = line.split(' ')
                label = class_name[int(content[0])]  # 标签
                for index in range(1, len(content)):
                    if index % 2 == 1:  # 下标为奇数，对应横坐标
                        x = (float(content[index])) * w
                        point_list.append(x)
                    else:  # 下标为偶数，对应纵坐标
                        y = (float(content[index])) * h
                        point_list.append(y)
                point_list = [point_list[i:i+2] for i in range(0, len(point_list), 2)]
                shape = {
                    'label': label,
                    'points': point_list,
                    'group_id': None,
                    'description': None,
                    'shape_type': 'polygon',
                    'flags': {},
                    'mask': None
                }
                labelme_json['shapes'].append(shape)
            json_name = txt_name.split('.')[0] + '.json'
            json_name_path = os.path.join(output_dir, json_name)
            fd = open(json_name_path, 'w')
            json.dump(labelme_json, fd, indent=4)
            fd.close()
            print("save json={}".format(json_name_path))

if __name__ == '__main__':
    txt_path = r'../test_txt2json/txt'
    image_path = r'../test_txt2json'
    output_dir = r'../test_txt2json'
    # 标签列表
    class_name = ['box_top', 'tape', 'tape_defect', 'box_bottom']  # 标签类别名
    convert_txt_to_labelme_json(txt_path, image_path, output_dir, class_name)

