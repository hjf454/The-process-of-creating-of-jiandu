import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import argparse


def enhance_image(image_path, output_path, sharpen_strength=1.5, contrast_strength=1.2, denoise_strength=1):
    """
    增强失真图片的主函数

    参数:
        image_path: 输入图片路径
        output_path: 输出图片路径
        sharpen_strength: 锐化强度（>1增强，建议1.2-2.0）
        contrast_strength: 对比度增强强度（>1增强，建议1.1-1.5）
        denoise_strength: 降噪强度（1-3，数值越大降噪越强）
    """
    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 转换为RGB模式（处理可能的透明通道）
            if img.mode in ('RGBA', 'LA'):
                background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                background.paste(img, img.split()[-1])
                img = background
            elif img.mode == 'P':
                img = img.convert('RGB')

            # 1. 降噪处理（先降噪再增强，避免放大噪声）
            if denoise_strength >= 1:
                # 轻度降噪
                img = img.filter(ImageFilter.MedianFilter(size=3 if denoise_strength == 1 else 5))

            # 2. 锐化处理（改善模糊）
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpen_strength)

            # 3. 对比度增强（增强细节层次）
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_strength)

            # 4. 轻微亮度调整（可选，根据图片情况）
            # enhancer = ImageEnhance.Brightness(img)
            # img = enhancer.enhance(1.05)  # 轻微提高亮度

            # 5. 细节增强（使用自定义卷积核强化边缘）
            if sharpen_strength > 1.3:
                kernel = np.array([
                    [-1, -1, -1],
                    [-1, 9 + (sharpen_strength - 1), -1],
                    [-1, -1, -1]
                ])
                img = img.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=3))

            # 保存处理后的图片
            img.save(output_path)
            print(f"成功处理: {os.path.basename(image_path)}")
            return True

    except Exception as e:
        print(f"处理失败 {os.path.basename(image_path)}: {str(e)}")
        return False


def process_folder(input_folder, output_folder, **kwargs):
    """批量处理文件夹中的所有图片"""
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # 遍历输入文件夹
    for filename in os.listdir(input_folder):
        # 检查文件格式
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            enhance_image(input_path, output_path, **kwargs)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='增强放大后失真的图片')
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出图片文件夹路径')
    parser.add_argument('--sharpen', type=float, default=1.5, help='锐化强度（建议1.2-2.0）')
    parser.add_argument('--contrast', type=float, default=1.2, help='对比度强度（建议1.1-1.5）')
    parser.add_argument('--denoise', type=int, default=1, help='降噪强度（1-3）')

    args = parser.parse_args()

    # 调用批量处理函数
    process_folder(
        input_folder=args.input,
        output_folder=args.output,
        sharpen_strength=args.sharpen,
        contrast_strength=args.contrast,
        denoise_strength=args.denoise
    )

    print("所有图片处理完成！")