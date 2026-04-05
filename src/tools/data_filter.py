#!/usr/bin/env python3
"""
数据清洗工具：过滤纯背景切片 + Z轴连续性验证
"""
import os
import sys
import argparse
import re
from pathlib import Path
import pydicom
from natsort import natsorted


def filter_background_slices(data_root):
    """删除纯背景切片（掩膜全为0）"""
    deleted_count = 0

    for patient_dir in sorted(Path(data_root).iterdir()):
        if not patient_dir.is_dir():
            continue

        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir() or study_dir.name == "mask":
                continue

            mask_dir = study_dir / "mask"
            if not mask_dir.exists():
                continue

            for mask_file in mask_dir.glob("mask_*.dcm"):
                try:
                    mask_data = pydicom.dcmread(mask_file).pixel_array
                    if mask_data.max() == 0:
                        # 提取序号
                        match = re.search(r'mask_(\d+)\.dcm', mask_file.name)
                        if match:
                            num = match.group(1)
                            img_file = study_dir / f"{num}.DCM"

                            # 删除图像和掩膜
                            if img_file.exists():
                                img_file.unlink()
                            mask_file.unlink()
                            deleted_count += 1
                            print(f"  删除纯背景切片: {img_file.name}")
                except Exception as e:
                    print(f"  错误处理 {mask_file}: {e}")

    return deleted_count


def filter_discontinuous_slices(data_root, gap_threshold=5):
    """删除Z轴不连续的切片段"""
    deleted_count = 0

    for patient_dir in sorted(Path(data_root).iterdir()):
        if not patient_dir.is_dir():
            continue

        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir() or study_dir.name == "mask":
                continue

            # 提取所有切片序号
            img_files = [f for f in study_dir.iterdir()
                        if f.suffix.upper() == '.DCM' and not f.name.startswith('mask')]

            if len(img_files) < 2:
                continue

            img_numbers = []
            for f in img_files:
                match = re.search(r'(\d+)\.DCM$', f.name, re.IGNORECASE)
                if match:
                    img_numbers.append(int(match.group(1)))

            if len(img_numbers) < 2:
                continue

            img_numbers_sorted = sorted(img_numbers)

            # 检测不连续点
            gaps = []
            for i in range(1, len(img_numbers_sorted)):
                gap = img_numbers_sorted[i] - img_numbers_sorted[i-1]
                if gap > gap_threshold:
                    gaps.append(i)

            if not gaps:
                continue

            # 保留最长连续段
            segments = []
            start = 0
            for gap_idx in gaps:
                segments.append((start, gap_idx))
                start = gap_idx
            segments.append((start, len(img_numbers_sorted)))

            longest_seg = max(segments, key=lambda x: x[1] - x[0])
            keep_numbers = set(img_numbers_sorted[longest_seg[0]:longest_seg[1]])

            # 删除不连续段
            for num in img_numbers:
                if num not in keep_numbers:
                    img_file = study_dir / f"{num}.DCM"
                    mask_file = study_dir / "mask" / f"mask_{num}.dcm"

                    if img_file.exists():
                        img_file.unlink()
                        deleted_count += 1
                    if mask_file.exists():
                        mask_file.unlink()

                    print(f"  删除不连续切片: {img_file.name}")

    return deleted_count


def main():
    parser = argparse.ArgumentParser(description='数据清洗：过滤无效切片')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录（Patient/Study/DICOM结构）')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['background', 'discontinuous', 'both'],
                        help='清洗模式（默认：both）')
    parser.add_argument('--gap_threshold', type=int, default=5,
                        help='Z轴不连续阈值（默认：5）')

    args = parser.parse_args()

    if not Path(args.data_root).exists():
        print(f"错误：数据目录不存在: {args.data_root}")
        sys.exit(1)

    print(f"开始数据清洗: {args.data_root}")
    print(f"模式: {args.mode}\n")

    total_deleted = 0

    if args.mode in ['background', 'both']:
        print("=" * 50)
        print("步骤1：过滤纯背景切片")
        print("=" * 50)
        count = filter_background_slices(args.data_root)
        total_deleted += count
        print(f"✓ 删除纯背景切片: {count} 个\n")

    if args.mode in ['discontinuous', 'both']:
        print("=" * 50)
        print("步骤2：过滤Z轴不连续切片")
        print("=" * 50)
        count = filter_discontinuous_slices(args.data_root, args.gap_threshold)
        total_deleted += count
        print(f"✓ 删除不连续切片: {count} 个\n")

    print("=" * 50)
    print(f"数据清洗完成！共删除 {total_deleted} 个切片")
    print("=" * 50)


if __name__ == '__main__':
    main()
