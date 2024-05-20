# 回放：相应的缺陷检测功能
from utils.box_utils import draw_by_json
import json

# 从json文件加载胶带缺陷检测结果
def load_tape_defect_result(img_path, json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    percentage, box_side_length = draw_by_json(data, img_path)  # 调用方法显示图片
    # print("胶带褶皱程度：{:.2f}%".format(percentage))
    # print("四段距离为：", box_side_length)
    return percentage, box_side_length

