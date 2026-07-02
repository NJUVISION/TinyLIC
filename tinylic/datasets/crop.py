import os
from PIL import Image
import math

# 输入输出路径
input_dir = "./jpegai_test"   # 原始图像文件夹
output_dir = "./jpegai_test_chunk"         # 切分后图像保存路径
os.makedirs(output_dir, exist_ok=True)

# 分辨率阈值
WIDTH_THRESHOLD = 1920
HEIGHT_THRESHOLD = 1080

def split_image(image_path):
    # 读取图像
    img = Image.open(image_path)
    w, h = img.size

    # 原文件名拆分
    filename = os.path.basename(image_path)
    name_parts = filename.split("_")
    base_idx = name_parts[0]  # 比如 '00032'
    suffix_parts = name_parts[1:]  # 比如 ['TE', '7680x5120', '8bit', 'sRGB.png']

    # 查找原本的分辨率部分
    resolution_idx = None
    for i, part in enumerate(suffix_parts):
        if 'x' in part and part.replace('x','').isdigit():
            resolution_idx = i
            break
    if resolution_idx is None:
        raise ValueError(f"无法识别图像分辨率：{filename}")

    # 计算切分份数
    num_splits_w = max(1, math.ceil(w / WIDTH_THRESHOLD))
    num_splits_h = max(1, math.ceil(h / HEIGHT_THRESHOLD))

    images_to_save = []
    idx = 0

    # 计算每块尺寸，按均分原则
    w_splits = [ (i * w // num_splits_w, (i+1) * w // num_splits_w) for i in range(num_splits_w) ]
    h_splits = [ (i * h // num_splits_h, (i+1) * h // num_splits_h) for i in range(num_splits_h) ]

    for hi, (y0, y1) in enumerate(h_splits):
        for wi, (x0, x1) in enumerate(w_splits):
            cropped = img.crop((x0, y0, x1, y1))
            new_w, new_h = cropped.size
            new_suffix_parts = suffix_parts.copy()
            new_suffix_parts[resolution_idx] = f"{new_w}x{new_h}"
            new_filename = f"{base_idx}_{idx}_{'_'.join(new_suffix_parts)}"
            images_to_save.append((cropped, new_filename))
            idx += 1

    # 保存
    for im, fname in images_to_save:
        save_path = os.path.join(output_dir, fname)
        im.save(save_path, "PNG")

# 遍历目录
for file in os.listdir(input_dir):
    if file.lower().endswith(".png"):
        split_image(os.path.join(input_dir, file))

print("切分完成！")
