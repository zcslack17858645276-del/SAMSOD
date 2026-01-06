import cv2
import os
import numpy as np


def save_mask_edge(
    mask: np.ndarray,
    save_path: str,
    kernel_size: int = 3
):
    """
    从 binary mask 生成边缘图并保存

    Args:
        mask: HxW, 0/255 或 0/1
        save_path: 输出路径（.png）
        kernel_size: 形态学核大小
    """

    # 确保是 uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # 若是 0/1，转成 0/255
    if mask.max() == 1:
        mask = mask * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 形态学梯度 = 膨胀 - 腐蚀
    edge = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        kernel
    )

    cv2.imwrite(save_path, edge)

def main():
    mask_path = "dataset/DUTS-TR/gt/ILSVRC2012_test_00000004.png"
    output_dir = "result/pseudo_labels"

    os.makedirs(output_dir, exist_ok=True)

    # 读取 mask（单通道）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert mask is not None, f"Failed to load mask: {mask_path}"

    save_path = os.path.join(output_dir, "pseudo_mask_edge.png")

    save_mask_edge(
        mask=mask,
        save_path=save_path,
        kernel_size=3
    )

    print(f"[DONE] Edge map saved to: {save_path}")


if __name__ == "__main__":
    main()