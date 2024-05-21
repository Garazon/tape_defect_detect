# 用于回放的工具
from PIL import Image
from matplotlib import pyplot as plt


# 根据字典形式绘制边界框
def draw_by_json(data, img_path):
    """
    :param data: 标注信息列表，字典形式
    :param image_path: 在哪张图上绘制
    :return:
    """

    # 读取图像
    image = Image.open(img_path)
    plt.imshow(image)
    # 设置颜色列表
    colors = [
        "#FF3838", "#520085", "#CB38FF", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A", "#3DDB86", "#00D4BB",
        "#344593", "#6473FF", "#0018EC", "#8438FF", "#FF95C8", "#FF37C7", "#2C99A8", "#00C2FF", "#1A9334", "#92CC17"
    ]
    # 绘制图像
    for item in data['data']:
        x = item["segments"]["x"]  # 获取掩码框边界点的横坐标
        y = item["segments"]["y"]  # 获取掩码框边界点的纵坐标
        plt.plot(x, y, color=colors[item["class"]], linewidth=2)  # 在图像上绘制掩码框边界
        plt.plot([x[-1], x[0]], [y[-1], y[0]], color=colors[item["class"]], linewidth=2)  # 将第一个点和最后一个点连接起来
    plt.show()  # 显示图像

    deviation_degree, defect_degree = [], []
    for item in data["display"]:
        if item["name"] == "deviation_degree":
            deviation_degree = item["value"]
        if item["name"] == "defect_degree":
            defect_degree = item["value"]

    return deviation_degree, defect_degree