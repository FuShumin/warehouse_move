# -*- coding: utf-8 -*-

import gradio as gr
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# 加载 YOLO 模型
# model = YOLO('./runs/detect/train6/weights/best.pt')
model = YOLO('/Users/apple/Downloads/train8/weights/best.pt')
label_info = {
    0: '双喜(硬经典)',
    1: '双喜(硬经典1906)',
    2: '双喜(花悦)',
    3: '双喜（莲香）',
    4: '双喜(软经典)'
}


def predict(inp):
    # 使用 YOLO 进行预测，并保存结果
    results = model(inp, save=True, conf=0.5)

    # 获取保存目录
    save_dir = results[0].save_dir

    # 生成保存的预测结果图像的完整路径
    save_path = os.path.join(save_dir, 'image0.jpg')  # 这里你可能需要根据实际情况修改文件名

    # 读取保存的预测结果图像，并返回
    output_image = Image.open(save_path)
    return output_image


iface = gr.Interface(
    fn=predict,  # 预测函数
    inputs=gr.inputs.Image(type="pil", label="上传你要检测的图片"),  # 输入类型和标签
    outputs=gr.outputs.Image(type="pil", label="预测结果"),  # 输出类型和标签
    description="标签信息：<br>0-双喜(硬经典)<br>1-双喜(硬经典1906)<br>2-双喜(花悦)<br>3-双喜（莲香）<br>4-双喜(软经典)"
)


iface.launch()
