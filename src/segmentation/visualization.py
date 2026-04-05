"""
可视化模块
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from natsort import natsorted
import math


def visualize_segmentation_steps(windowed_img, seed_mask, growth_mask, largest_mask,
                                  optimized_mask, tibia_mask, fibula_mask, output_path):
    """
    六宫格可视化分割步骤

    Args:
        windowed_img: 预处理图像
        seed_mask: 种子点掩膜
        growth_mask: 区域生长结果
        largest_mask: 最大连通域
        optimized_mask: 形态学优化结果（可为None）
        tibia_mask: 胫骨掩膜
        fibula_mask: 腓骨掩膜
        output_path: 输出路径
    """
    plt.figure(figsize=(16, 10))

    titles = [
        "Preprocessed Image",
        "Kmeans Seed Points",
        "Region Growing Result",
        "Largest Connected Components",
        "Morphological Optimization",
        "Final Segmentation (Red: Tibia, Green: Fibula)"
    ]

    plt.subplot(231)
    plt.imshow(windowed_img, cmap='gray')
    plt.title(titles[0])
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(seed_mask, cmap='gray')
    plt.title(titles[1])
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(growth_mask, cmap='gray')
    plt.title(titles[2])
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(largest_mask, cmap='gray')
    plt.title(titles[3])
    plt.axis('off')

    plt.subplot(235)
    if optimized_mask is not None:
        plt.imshow(optimized_mask, cmap='gray')
    else:
        plt.imshow(largest_mask, cmap='gray')
    plt.title(titles[4])
    plt.axis('off')

    # 彩色叠加
    color_result = cv2.cvtColor(windowed_img, cv2.COLOR_GRAY2BGR)
    color_result[tibia_mask > 0] = [0, 0, 255]
    color_result[fibula_mask > 0] = [0, 255, 0]
    plt.subplot(236)
    plt.imshow(cv2.cvtColor(color_result, cv2.COLOR_BGR2RGB))
    plt.title(titles[5])
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()


def create_all_visual_summary(patient_output_dir):
    """为每个患者创建所有可视化结果的汇总图"""
    png_files = [f for f in os.listdir(patient_output_dir)
                 if f.endswith('.png') and f != 'AllVisual.png']

    if not png_files:
        return None

    png_files = natsorted(png_files)
    n_images = len(png_files)
    n_cols = min(4, n_images)
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 6))

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, png_file in enumerate(png_files):
        row, col = idx // n_cols, idx % n_cols
        img = Image.open(os.path.join(patient_output_dir, png_file))
        axes[row, col].imshow(img)
        axes[row, col].set_title(png_file, fontsize=10)
        axes[row, col].axis('off')

    for idx in range(n_images, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    all_visual_path = os.path.join(patient_output_dir, 'AllVisual.png')
    plt.savefig(all_visual_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()

    return all_visual_path
