# 胫腓骨 CT 无监督分割 — 项目简述

> 路径：`/data2/lyb/ankleSeg/TibiaFibula-CT-Seg-Unsupervised`  
> 更新：2026-04-05

---

## 1. 项目是做什么的

针对踝关节 CT（DICOM）切片，对**胫骨 / 腓骨**做自动二值分割，**不依赖人工标注**：K-means 取骨种子 + 区域生长 + 连通域与形态学后处理，并支持**反向扫描**下的胫腓左右判别（`separation_strategy=enhanced`）。输出为与原始 DICOM 空间对齐的掩膜（背景 0、胫骨 1、腓骨 2），供下游监督学习或质检使用。

---

## 2. 快速开始

### 依赖

```bash
pip install pydicom numpy opencv-python scikit-learn matplotlib natsort pillow
```

### 批量分割（主入口）

```bash
python src/pipelines/generate_masks.py \
  --input_dir /path/to/data \
  --output_dir ./outputs \
  --window_center 300 \
  --window_width 800 \
  --growth_threshold 200 \
  --separation_strategy enhanced
```

默认会保存 DICOM 掩膜与 PNG 可视化（可用 `--no_morphology` 做消融对照）。

### 单张调试

```bash
python src/pipelines/single_debug.py \
  --input /path/to/sample.DCM \
  --output ./debug.png
```

### 数据清洗（原地删无效切片）

```bash
python src/tools/data_filter.py \
  --data_root /path/to/data_with_mask \
  --mode both \
  --gap_threshold 5
```

`--mode`：`background`（纯背景掩膜）、`discontinuous`（Z 轴序号跳跃）、`both`。

---

## 3. 仓库结构（当前）

```
TibiaFibula-CT-Seg-Unsupervised/
├── src/
│   ├── segmentation/       # core.py / io_dicom.py / visualization.py
│   ├── pipelines/          # generate_masks.py / single_debug.py
│   └── tools/              # data_filter.py
├── Data_new/               # 反脚等子集输入（trainset / testset）
├── Data_new_AfterSeg/      # 已跑通的掩膜结果（供下游训练）⭐
├── outputs/                # 新管线默认输出（visualization/ + masks/）
├── datavisual.py           # PyTorch Dataset，对接下游训练
├── deleteUnusedFolder.py   # 目录清理脚本
├── archive_old_versions/   # 旧脚本与历史数据说明
└── PROJECT_REPORT.md       # 本文件
```

---

## 4. 代码模块职责

| 位置 | 作用 |
|------|------|
| `src/segmentation/core.py` | 预处理、K-means 种子、区域生长、最大连通域、形态学、`basic`/`enhanced` 胫腓分离 |
| `src/segmentation/io_dicom.py` | DICOM 读写、掩膜写回与空间元数据对齐 |
| `src/segmentation/visualization.py` | 步骤可视化与患者级汇总图 |
| `src/pipelines/generate_masks.py` | 批量 CLI：窗宽窗位、生长阈值、`--no_morphology`、`--separation_strategy` 等 |
| `src/pipelines/single_debug.py` | 单张调参与六宫格输出 |
| `src/tools/data_filter.py` | 合并原 `screenJingFei1/2` 的清洗逻辑 |
| `datavisual.py` | `ANKLEDicomDataset`：配对加载、窗位归一化、同步增强、伪 RGB |

---

## 5. 重构说明（与旧脚本对应）

代码已模块化：**公共算法集中在 `src/segmentation/`，路径与参数全部 CLI 化**，去掉多文件重复实现（原约 800 行重复逻辑收敛为可维护模块）。

| 旧文件（多在 `archive_old_versions/`） | 现用法 |
|----------------------------------------|--------|
| `segJingFei_batch.py`、`MaskGenerate.py`、`makeGenerate_2.py` | `generate_masks.py` + `core.py`（增强分离即 `enhanced`） |
| `segJingFei_Single.py` | `single_debug.py` |
| `segJingFei_batch_deletemorphology.py` | `generate_masks.py --no_morphology` |
| `screenJingFei1/2_*.py` | `data_filter.py --mode ...` |

---

## 6. 数据与输出

- **输入**：患者 → Study → 若干 `.DCM`；反脚子集建议使用 `Data_new/`，并配合 `--separation_strategy enhanced`。
- **批量输出**（`--output_dir`，默认 `./outputs`）：
  - `visualization/<患者>/`：步骤 PNG、`AllVisual.png`
  - `masks/<患者>/`：`mask_*.dcm`
- **历史大数据与旧可视化**：见 `archive_old_versions/` 内说明；当前仓库根目录以 `Data_new*` 与 `outputs/` 为主。

---

## 7. 算法流水线（摘要）

1. HU + 骨窗 + 高斯平滑 + 中心裁剪  
2. K-means（默认 5 类）取最亮两类作骨种子  
3. 四邻域 BFS 区域生长（阈值可调）  
4. 保留面积最大的两个连通域 → 胫腓分离（几何/增强策略）  
5. 可选 3×3 形态学闭开运算  
6. 360×360 结果填回 512×512 中心区域，写 DICOM 分割实例并复制空间标签  

更细的公式与参数讨论见学位论文或代码内实现。

---

## 8. 技术要点

- **无监督伪标签管线**：无需标注即可批量生成掩膜。  
- **DICOM 空间一致**：掩膜与原始切片可对齐叠加查看。  
- **反脚 / 扫描方向**：`enhanced` 策略用几何先验减轻左右翻转导致的标签交换问题。  
- **质量闭环**：`data_filter` 过滤空掩膜与 Z 轴断裂片。  
- **下游就绪**：`datavisual.py` 直接接 PyTorch 训练。

---

## 9. 后续可做（非必须）

单元测试、YAML 配置、日志与进度条、批量多进程加速等（原重构报告中的建议仍适用）。
