import os
import cv2
import numpy as np
import glob

def generate_pseudo_labels(
    image_path: str,
    mask_path: str,
    output_dir: str,
    prompt_mask_size: int = 256
):
    """
    输入一张图片和对应 mask，生成三种伪标签并保存为图片：
    1. 点提示（point）
    2. 框提示（box）
    3. 掩码提示（mask）
    """

    os.makedirs(output_dir, exist_ok=True)

    # ---------- 读取数据 ----------
    image = cv2.imread(image_path)
    assert image is not None, f"Failed to read image: {image_path}"

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert mask is not None, f"Failed to read mask: {mask_path}"

    h, w = mask.shape

    # ---------- 1. Point 伪标签 ----------
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    points, labels = _simulate_click_from_mask(mask)

    if labels[0] == 1:
        x, y = points[0].astype(int)
        cv2.circle(
            mask_vis,
            (x, y),
            radius=5,
            color=(0, 0, 255),  # 红色
            thickness=-1
        )

    cv2.imwrite(
        os.path.join(output_dir, "pseudo_mask_with_point.png"),
        mask_vis
    )

    # ---------- 2. Box 伪标签 ----------
    box = _simulate_box_from_mask(mask)

    x1, y1, x2, y2 = box[0].astype(int)
    if (x2 > x1) and (y2 > y1):
        # 红色框（BGR）
        cv2.rectangle(
            mask_vis,
            (x1, y1),
            (x2, y2),
            color=(0, 0, 255),  # 红色
            thickness=2
        )

    cv2.imwrite(
        os.path.join(output_dir, "pseudo_mask_with_box.png"),
        mask_vis
    )

    # ---------- 3. Mask 伪标签 ----------
    mask_prompt = _preprocess_mask_prompt(mask, prompt_mask_size)

    # 为了可视化，放大回原图大小
    mask_prompt_vis = cv2.resize(
        mask_prompt, (w, h), interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(
        os.path.join(output_dir, "pseudo_mask.png"),
        mask_prompt_vis
    )

    return {
        "point": points,
        "point_label": labels,
        "box": box,
        "mask_prompt": mask_prompt
    }


def _simulate_click_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)

    if len(y_indices) > 0:
        random_idx = np.random.randint(0, len(y_indices))
        x = x_indices[random_idx]
        y = y_indices[random_idx]

        points = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
    else:
        points = np.array([[0, 0]], dtype=np.float32)
        labels = np.array([-1], dtype=np.int32)

    return points, labels

def _simulate_box_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)

    if len(y_indices) > 0:
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        x_min = max(0, x_min - np.random.randint(0, 5))
        x_max = min(mask.shape[1] - 1, x_max + np.random.randint(0, 5))
        y_min = max(0, y_min - np.random.randint(0, 5))
        y_max = min(mask.shape[0] - 1, y_max + np.random.randint(0, 5))

        box = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
    else:
        box = np.array([[0, 0, 0, 0]], dtype=np.float32)

    return box

def _preprocess_mask_prompt(mask, prompt_mask_size):
    mask_low_res = cv2.resize(
        mask,
        (prompt_mask_size, prompt_mask_size),
        interpolation=cv2.INTER_NEAREST
    )

    if mask_low_res.max() > 0:
        prob = np.random.random()

        kernel_size = np.random.randint(3, 8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if prob < 0.4:
            mask_low_res = cv2.erode(mask_low_res, kernel, iterations=1)
        elif prob < 0.8:
            mask_low_res = cv2.dilate(mask_low_res, kernel, iterations=1)

    return mask_low_res

def main():
    # ======================
    # 1. 路径配置
    # ======================
    image_dir = "dataset/DUTS-TR/im"
    mask_dir = "dataset/DUTS-TR/gt"
    output_root = "result/pseudo_labels"

    os.makedirs(output_root, exist_ok=True)

    # ======================
    # 2. 读取数据列表
    # ======================
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    assert len(image_paths) == len(mask_paths), "image / mask 数量不一致"

    print(f"[INFO] Found {len(image_paths)} samples")

    # ======================
    # 3. 逐张生成伪标签
    # ======================
    index = 0
    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        index+=1
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(output_root, name)

        print(f"[{idx+1}/{len(image_paths)}] Processing {name}")

        generate_pseudo_labels(
            image_path=img_path,
            mask_path=mask_path,
            output_dir=out_dir,
            prompt_mask_size=256
        )
        if index == 1:
            break

    print("[DONE] Pseudo label generation finished.")


if __name__ == "__main__":
    main()