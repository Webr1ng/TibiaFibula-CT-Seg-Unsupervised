#!/usr/bin/env python3
"""
主批处理入口：批量生成胫腓骨分割掩膜
支持DICOM掩膜生成、PNG可视化、汇总图生成
"""
import os
import sys
import argparse
import re
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.core import (
    preprocess_image, get_seed_points, region_growth,
    keep_largest_connected_components, morphology_optimization,
    separate_tibia_fibula
)
from segmentation.io_dicom import load_dicom, create_dicom_mask
from segmentation.visualization import visualize_segmentation_steps, create_all_visual_summary
import numpy as np


def process_single_slice(dicom_path, vis_output_dir, mask_output_dir, filename, config):
    """处理单个DICOM切片"""
    try:
        # 1. 加载与预处理
        ds = load_dicom(dicom_path)
        windowed_img = preprocess_image(ds, config['window_center'], config['window_width'])

        # 2. K-means种子点
        seed_mask, seed_coords = get_seed_points(windowed_img, config['n_clusters'])

        # 3. 区域生长
        growth_mask = region_growth(windowed_img, seed_coords, config['growth_threshold'])

        # 4. 保留最大连通域
        largest_mask = keep_largest_connected_components(growth_mask, top_k=2)

        # 5. 形态学优化（可选）
        if config['use_morphology']:
            optimized_mask = morphology_optimization(largest_mask)
        else:
            optimized_mask = None

        # 6. 胫腓骨区分
        final_mask = optimized_mask if optimized_mask is not None else largest_mask
        try:
            tibia_mask, fibula_mask = separate_tibia_fibula(final_mask, config['separation_strategy'])
        except ValueError:
            tibia_mask = final_mask
            fibula_mask = np.zeros_like(final_mask)

        # 7. 保存DICOM掩膜到masks目录
        if config['save_mask']:
            mask_360 = np.zeros((360, 360), dtype=np.uint16)
            mask_360[tibia_mask > 0] = 1
            mask_360[fibula_mask > 0] = 2

            match = re.search(r'(\d+)\.DCM$', filename, re.IGNORECASE)
            mask_num = match.group(1) if match else str(len(list(Path(mask_output_dir).glob('mask_*.dcm'))) + 1)
            mask_path = Path(mask_output_dir) / f"mask_{mask_num}.dcm"

            create_dicom_mask(ds, mask_360, str(mask_path))

        # 8. 保存PNG可视化到visualization目录
        if config['save_png']:
            output_path = os.path.join(vis_output_dir, filename.replace('.DCM', '.png'))
            visualize_segmentation_steps(
                windowed_img, seed_mask, growth_mask, largest_mask,
                optimized_mask, tibia_mask, fibula_mask, output_path
            )

        return True

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False


def batch_process(input_dir, output_dir, config):
    """批量处理所有患者"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建visualization和masks两个子目录
    vis_root = output_path / "visualization"
    mask_root = output_path / "masks"
    vis_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    stats = {'total_patients': 0, 'total_studies': 0, 'total_samples': 0,
             'successful_samples': 0, 'failed_samples': 0}

    for patient_dir in sorted(input_path.iterdir()):
        if not patient_dir.is_dir():
            continue

        stats['total_patients'] += 1
        print(f"\nProcessing patient: {patient_dir.name}")

        # 为每个患者创建独立的visualization和masks子目录
        patient_vis_dir = vis_root / patient_dir.name
        patient_mask_dir = mask_root / patient_dir.name
        patient_vis_dir.mkdir(exist_ok=True)
        patient_mask_dir.mkdir(exist_ok=True)

        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir() or study_dir.name == "mask" or not re.search(r'\d', study_dir.name):
                continue

            stats['total_studies'] += 1
            img_files = natsorted([f for f in study_dir.iterdir()
                                   if f.suffix.upper() == '.DCM' and not f.name.startswith('mask')])

            print(f"  ├─ Processing study: {study_dir.name} ({len(img_files)} files)")

            for img_file in tqdm(img_files, desc=f"  ├─ {study_dir.name}", leave=False):
                success = process_single_slice(str(img_file), str(patient_vis_dir),
                                                str(patient_mask_dir), img_file.name, config)
                stats['total_samples'] += 1
                if success:
                    stats['successful_samples'] += 1
                else:
                    stats['failed_samples'] += 1

        # 生成汇总图
        if config['save_png']:
            create_all_visual_summary(str(patient_vis_dir))

    # 打印统计
    print("\n" + "="*50)
    print("Batch processing completed!")
    print(f"Total patients: {stats['total_patients']}")
    print(f"Total studies: {stats['total_studies']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful: {stats['successful_samples']}")
    print(f"Failed: {stats['failed_samples']}")
    if stats['total_samples'] > 0:
        print(f"Success rate: {stats['successful_samples']/stats['total_samples']*100:.2f}%")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='批量生成胫腓骨分割掩膜')
    parser.add_argument('--input_dir', type=str, default='',
                        help='输入数据根目录（Patient/Study/DICOM三级结构）')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录（默认：./outputs）')
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
    parser.add_argument('--save_mask', action='store_true', default=True,
                        help='保存DICOM掩膜（默认：True）')
    parser.add_argument('--save_png', action='store_true', default=True,
                        help='保存PNG可视化（默认：True）')

    args = parser.parse_args()

    if not args.input_dir:
        print("错误：必须指定 --input_dir 参数")
        print("\n使用示例：")
        print("  python generate_masks.py --input_dir /path/to/data --output_dir ./outputs")
        sys.exit(1)

    if not Path(args.input_dir).exists():
        print(f"错误：输入目录不存在: {args.input_dir}")
        sys.exit(1)

    config = {
        'window_center': args.window_center,
        'window_width': args.window_width,
        'growth_threshold': args.growth_threshold,
        'n_clusters': args.n_clusters,
        'use_morphology': not args.no_morphology,
        'separation_strategy': args.separation_strategy,
        'save_mask': args.save_mask,
        'save_png': args.save_png
    }

    batch_process(args.input_dir, args.output_dir, config)


if __name__ == '__main__':
    main()
