# 回放：相应的缺陷检测功能
from playback.utils import draw_by_json
import json

# 从json文件加载胶带缺陷检测结果
def load_tape_defect_result(img_path, json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    deviation_degree, defect_degree = draw_by_json(data["defect"], img_path)  # 调用方法显示图片
    # print("胶带褶皱程度：{:.2f}%".format(percentage))
    # print("四段距离为：", box_side_length)
    return deviation_degree, defect_degree

if __name__ == '__main__':
    img_path = r"../test_img/test_bottom.jpg"
    json_path = r"../test_img/test_bottom.json"
    deviation_degree, defect_degree = load_tape_defect_result(img_path, json_path)
    print(deviation_degree)
    print(defect_degree)

