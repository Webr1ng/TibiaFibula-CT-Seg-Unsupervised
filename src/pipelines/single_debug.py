#!/usr/bin/env python3
"""
单张DICOM调试入口：可视化完整分割流程
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.core import (
    preprocess_image, get_seed_points, region_growth,
    keep_largest_connected_components, morphology_optimization,
    separate_tibia_fibula
)
from segmentation.io_dicom import load_dicom
from segmentation.visualization import visualize_segmentation_steps
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='单张DICOM切片分割调试')
    parser.add_argument('--input', type=str, required=True,
                        help='输入DICOM文件路径')
    parser.add_argument('--output', type=str, default='./debug_output.png',
                        help='输出可视化图像路径（默认：./debug_output.png）')
    parser.add_argument('--window_center', type=int, default=300,
                        help='骨窗窗位（默认：300）')
    parser.add_argument('--window_width', type=int, default=800,
                        help='骨窗窗宽（默认：800）')
    parser.add_argument('--growth_threshold', type=int, default=200,
                        help='区域生长阈值（默认：200）')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='K-means聚类数（默认：5）')
    parser.add_argument('--no_morphology', action='store_true',
                        help='禁用形态学优化')
    parser.add_argument('--separation_strategy', type=str, default='enhanced',
                        choices=['basic', 'enhanced'],
                        help='胫腓骨区分策略（默认：enhanced）')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误：输入文件不存在: {args.input}")
        sys.exit(1)

    print(f"Processing: {args.input}")
    print(f"Parameters: WC={args.window_center}, WW={args.window_width}, "
          f"threshold={args.growth_threshold}, clusters={args.n_clusters}")

    # 1. 加载与预处理
    ds = load_dicom(args.input)
    windowed_img = preprocess_image(ds, args.window_center, args.window_width)

    # 2. K-means种子点
    seed_mask, seed_coords = get_seed_points(windowed_img, args.n_clusters)

    # 3. 区域生长
    growth_mask = region_growth(windowed_img, seed_coords, args.growth_threshold)

    # 4. 保留最大连通域
    largest_mask = keep_largest_connected_components(growth_mask, top_k=2)

    # 5. 形态学优化（可选）
    if not args.no_morphology:
        optimized_mask = morphology_optimization(largest_mask)
    else:
        optimized_mask = None

    # 6. 胫腓骨区分
    final_mask = optimized_mask if optimized_mask is not None else largest_mask
    try:
        tibia_mask, fibula_mask = separate_tibia_fibula(final_mask, args.separation_strategy)
    except ValueError as e:
        print(f"警告：胫腓骨区分失败 ({e})，使用单一掩膜")
        tibia_mask = final_mask
        fibula_mask = np.zeros_like(final_mask)

    # 7. 保存可视化
    visualize_segmentation_steps(
        windowed_img, seed_mask, growth_mask, largest_mask,
        optimized_mask, tibia_mask, fibula_mask, args.output
    )

    print(f"✓ 可视化结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
