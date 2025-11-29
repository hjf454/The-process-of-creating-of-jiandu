#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_jian_characters.py
------------------------
用于将简牍字符图片进行分割并保存成单独的字符图像。

依赖:
    pip install opencv-python numpy

用法:
    python split_jian_characters.py --input ./jian_full/img.png --output output_chars
"""

import cv2
import numpy as np
import os
import argparse

def split_jian_characters(input_image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取彩色图（原图用于裁剪）
    img_color = cv2.imread(input_image_path)
    if img_color is None:
        raise FileNotFoundError(f"无法读取图片: {input_image_path}")

    # 灰度化用于分割
    gray_for_detect = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_for_detect, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 膨胀连接偏旁部件
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    merged = cv2.dilate(clean, merge_kernel, iterations=1)

    # 找轮廓（合并后的）
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        if h < 20:
            continue

        # 裁剪原彩色图的对应区域
        char_color = img_color[y:y+h, x:x+w]
        char_gray = cv2.cvtColor(char_color, cv2.COLOR_BGR2GRAY)

        # 对比度增强
        char_gray = cv2.equalizeHist(char_gray)

        # 放大
        scale = 3
        char_gray = cv2.resize(char_gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

        save_path = os.path.join(output_folder, f"char_{count+1}.png")
        cv2.imwrite(save_path, char_gray)
        count += 1

    print(f"切分完成，共保存 {count} 个字符图像。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简牍字符切分工具")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", default="output_chars", help="输出目录")
    args = parser.parse_args()

    split_jian_characters(args.input, args.output)