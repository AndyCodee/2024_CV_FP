import os
from PIL import Image

base_path = "C:/Users/ouche/Desktop/CV_FP/train"

augmentations = {
    "_left5": lambda img: img.rotate(5, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_left10": lambda img: img.rotate(10, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_right5": lambda img: img.rotate(-5, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_right10": lambda img: img.rotate(-10, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_mirror": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    "_left5m": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(5, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_left10m": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(10, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_right5m": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-5, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_right10m": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-10, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_zoom_in": lambda img: zoom_in(img, scale=1.2),  # 方法1：放大並切割
    "_zoom_out": lambda img: zoom_out(img, scale=0.8)  # 方法2：縮小並填充黑色
}

def zoom_in(img, scale=1.2):
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height

    return img_resized.crop((left, top, right, bottom))

def zoom_out(img, scale=0.8):
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (width, height), (0, 0, 0))
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    new_img.paste(img_resized, (left, top))

    return new_img

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.isdir(folder_path):
        print(f"{folder_path} 不是資料夾，跳過。")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"無法讀取圖片 {img_path}，錯誤：{e}")
            continue

        for aug_suffix, aug_function in augmentations.items():
            aug_img = aug_function(img)  
            original_width, original_height = img.size
            new_width, new_height = aug_img.size

            left = (new_width - original_width) // 2
            top = (new_height - original_height) // 2
            right = left + original_width
            bottom = top + original_height

            aug_img = aug_img.crop((left, top, right, bottom))
            aug_img_path = os.path.join(folder_path, f"{os.path.splitext(img_name)[0]}{aug_suffix}{os.path.splitext(img_name)[1]}")
            aug_img.save(aug_img_path)

        print(f"已處理圖片：{img_path}")

print("所有圖片擴增完成！")