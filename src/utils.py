import cv2
import os
import numpy as np


# ========== 批量处理核心工具函数 ==========
def get_image_list(folder_path, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """获取指定目录下的所有图片文件路径"""
    image_paths = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return image_paths

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in extensions:
            image_paths.append(os.path.abspath(file_path))
    return image_paths


def get_unique_save_path(original_img_path, save_dir, suffix, ext='.jpg'):
    """生成唯一的保存路径（按原图片名区分）"""
    img_name = os.path.splitext(os.path.basename(original_img_path))[0]
    return os.path.join(save_dir, f"{img_name}{suffix}{ext}")


# ========== 核心图像处理与坐标映射 ==========
def resize_image(img, scale=1.0):
    """
    简化缩放逻辑：直接按比例缩放，不强制固定画布尺寸
    解决维度不匹配问题
    """
    h, w = img.shape[:2]
    # 计算新尺寸（强制整数）
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 边界检查：避免缩放后尺寸为0
    new_w = max(10, new_w)
    new_h = max(10, new_h)

    # 缩放图像
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 返回缩放后图像和实际缩放比例（用于坐标映射）
    actual_scale_w = new_w / w
    actual_scale_h = new_h / h

    return resized_img, actual_scale_w, actual_scale_h


def pixel_to_original(x, y, scale_w, scale_h):
    """窗口坐标转换为原图坐标（简化版，无偏移）"""
    orig_x = int(x / scale_w)
    orig_y = int(y / scale_h)
    return orig_x, orig_y


def original_to_pixel(x, y, scale_w, scale_h):
    """原图坐标转换为窗口坐标（简化版，无偏移）"""
    disp_x = int(x * scale_w)
    disp_y = int(y * scale_h)
    return disp_x, disp_y


def get_histogram_threshold(gray_img, roi_rect):
    """
    修复：返回阈值数值而非图像
    对指定ROI区域计算直方图，自动获取二值化阈值（OTSU算法）
    """
    x1, y1, x2, y2 = roi_rect
    # 裁剪ROI区域（严格边界检查）
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(gray_img.shape[1] - 1, x2)
    y2 = min(gray_img.shape[0] - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 127  # 默认阈值

    roi = gray_img[y1:y2, x1:x2]
    # OTSU自动阈值（返回阈值数值）
    otsu_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_thresh


def get_largest_connected_component(mask):
    """
    提取掩膜中的最大连通分量（核心新增函数）
    :param mask: 二值化掩膜（bool类型）
    :return: 只保留最大连通分量的掩膜
    """
    # 转换为uint8格式（OpenCV连通域分析要求）
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 查找所有连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    # 如果只有背景（无连通域），返回原掩膜
    if num_labels <= 1:
        return mask

    # 排除背景（label=0），找到面积最大的连通域
    max_area = 0
    max_label = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = label

    # 只保留最大连通分量
    largest_component_mask = (labels == max_label)

    return largest_component_mask


def roi_binary_mask(gray_img, roi_rect, threshold):
    """
    对指定ROI区域生成二值化掩膜（新增最大连通分量提取）
    """
    x1, y1, x2, y2 = roi_rect
    mask = np.zeros_like(gray_img, dtype=np.bool_)

    # 边界检查
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(gray_img.shape[1] - 1, x2)
    y2 = min(gray_img.shape[0] - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return mask

    # ROI区域二值化（修复阈值参数类型）
    roi = gray_img[y1:y2, x1:x2]
    _, binary_roi = cv2.threshold(roi, float(threshold), 255, cv2.THRESH_BINARY_INV)
    mask[y1:y2, x1:x2] = binary_roi > 0

    # 核心修改：提取最大连通分量，剔除噪声
    mask = get_largest_connected_component(mask)

    return mask


def calculate_centroid(mask):
    """计算掩膜重心坐标（亚像素级）"""
    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)
    return (cx, cy)


# ========== 文件操作函数 ==========
def write_to_txt(save_path, point_id, cx, cy, mode="a"):
    """保存控制点坐标到TXT"""
    with open(save_path, mode, encoding="utf-8") as f:
        f.write(f"{point_id}, {cx:.4f}, {cy:.4f}\n")


def init_txt(save_path):
    """初始化TXT文件表头"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("控制点编号, 像点x坐标, 像点y坐标\n")