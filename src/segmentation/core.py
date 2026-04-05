"""
核心分割算法模块
包含：预处理、K-means种子点、区域生长、连通域分析、形态学、胫腓骨区分
"""
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import deque


def preprocess_image(dicom_ds, window_center=250, window_width=1000, crop_size=360):
    """
    预处理CT的DICOM数据：HU值转换、骨窗映射、去噪、裁剪

    Args:
        dicom_ds: pydicom Dataset对象
        window_center: 窗位
        window_width: 窗宽
        crop_size: 裁剪后尺寸

    Returns:
        windowed_img: 预处理后的图像 (crop_size, crop_size)
    """
    # HU值转换
    slope = getattr(dicom_ds, "RescaleSlope", 1.0)
    intercept = getattr(dicom_ds, "RescaleIntercept", 0.0)
    hu_image = dicom_ds.pixel_array.astype(np.float32) * slope + intercept

    # 骨窗映射
    min_val = window_center - window_width / 2.0
    max_val = window_center + window_width / 2.0
    windowed_img = np.clip(hu_image, min_val, max_val)
    windowed_img = (windowed_img - min_val) / (max_val - min_val) * 255.0
    windowed_img = windowed_img.astype(np.uint8)

    # 高斯去噪
    windowed_img = cv2.GaussianBlur(windowed_img, (5, 5), 0)

    # 中心裁剪
    h, w = windowed_img.shape
    if h == 512 and w == 512:
        start = (512 - crop_size) // 2
        return windowed_img[start:start+crop_size, start:start+crop_size]
    else:
        return cv2.resize(windowed_img, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)


def get_seed_points(windowed_img, n_clusters=5):
    """
    K-means聚类提取骨骼种子点

    Args:
        windowed_img: 预处理后的图像
        n_clusters: 聚类数量

    Returns:
        seed_mask: 种子点掩膜
        seed_coords: 种子点坐标列表 [(y,x), ...]
    """
    h, w = windowed_img.shape
    pixels = windowed_img.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(cluster_centers)[::-1]
    bone_cluster_ids = sorted_idx[:2]  # 取灰度最高的前2簇

    seed_mask = np.isin(labels.reshape(h, w), bone_cluster_ids).astype(np.uint8)
    seed_coords = np.argwhere(seed_mask == 1)

    return seed_mask, seed_coords


def region_growth(img, seed_coords, threshold=20):
    """
    区域生长算法（BFS四邻域）

    Args:
        img: 灰度图像
        seed_coords: 种子点坐标
        threshold: 灰度差阈值

    Returns:
        growth_mask: 生长后的掩膜
    """
    h, w = img.shape
    growth_mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)

    queue = deque()
    for (y, x) in seed_coords:
        if not visited[y, x]:
            queue.append((y, x))
            visited[y, x] = True
            growth_mask[y, x] = 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        y, x = queue.popleft()
        current_gray = img[y, x]

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if abs(img[ny, nx] - current_gray) <= threshold:
                    visited[ny, nx] = True
                    growth_mask[ny, nx] = 1
                    queue.append((ny, nx))

    return growth_mask


def keep_largest_connected_components(mask, top_k=2):
    """保留最大的k个连通域"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    result_mask = np.zeros_like(mask)
    for cnt in contours_sorted[:top_k]:
        cv2.drawContours(result_mask, [cnt], -1, 255, -1)
    return result_mask


def morphology_optimization(mask, kernel_size=3):
    """形态学后处理：闭运算+开运算"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def separate_tibia_fibula(mask, strategy='enhanced'):
    """
    区分胫骨和腓骨

    Args:
        mask: 包含两个连通域的掩膜
        strategy: 'basic' 或 'enhanced'

    Returns:
        tibia_mask, fibula_mask
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("Not enough connected regions")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if strategy == 'enhanced':
        # 增强版：面积判断——胫骨面积始终大于腓骨，与扫描方向无关
        tibia_contour, fibula_contour = contours[0], contours[1]
    else:
        # 基础版：仅用质心x坐标判断
        tibia_center = cv2.moments(contours[0])
        fibula_center = cv2.moments(contours[1])
        tibia_x = tibia_center['m10'] / tibia_center['m00'] if tibia_center['m00'] != 0 else 0
        fibula_x = fibula_center['m10'] / fibula_center['m00'] if fibula_center['m00'] != 0 else 0

        if fibula_x < tibia_x:
            tibia_contour, fibula_contour = contours[1], contours[0]
        else:
            tibia_contour, fibula_contour = contours[0], contours[1]

    tibia_mask = np.zeros_like(mask)
    fibula_mask = np.zeros_like(mask)
    cv2.drawContours(tibia_mask, [tibia_contour], -1, 255, -1)
    cv2.drawContours(fibula_mask, [fibula_contour], -1, 255, -1)

    return tibia_mask, fibula_mask
