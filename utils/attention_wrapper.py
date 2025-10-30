import torch
import os
from utils.logger import logging_info
from utils.attention_visualizer import compute_attention_masks_with_vis
from skimage import measure
from torchvision.transforms import functional as TF

def compute_attention_masks(pred_crop0, img_free_crop=None, 
                           temperature=0.1, blur_sigma=0.5, attn_defect_accum=None,
                           center_weight=0.7, radial_decay=0.8, connected_components=True, min_area_ratio=0.01,
                           center_region_ratio=0.3, adaptive_threshold=True, threshold_factor=0.5):
    """
    计算注意力掩码，基于像素差异，并可选择性地过滤掉背景区域的错误高注意力信号
    
    参数:
        pred_crop0: 低分辨率缺陷模型的预测结果
        img_free_crop: 真实无缺陷背景对应的裁剪区域，用于过滤
        temperature: 温度系数，用于控制注意力图的锐度
        blur_sigma: 高斯模糊的标准差，用于平滑注意力图
        attn_defect_accum: 累积的缺陷注意力掩码，如果提供则使用EMA更新
        center_weight: 中心加权的权重参数 (0-1)，越大则中心区域权重越高
        radial_decay: 径向衰减率 (0-1)，越大则权重向边缘衰减越慢
        connected_components: 是否启用连通域分析过滤
        min_area_ratio: 连通域最小面积比例，小于此值将被过滤
        center_region_ratio: 图像中心区域的大小比例 (0-1)，用于定义中心区域
        adaptive_threshold: 是否使用自适应阈值而非固定值
        threshold_factor: 自适应阈值时的缩放因子 (0-1)
        
    返回:
        attn_defect: 缺陷区域注意力掩码 [0,1]
        attn_bg: 背景区域注意力掩码 [0,1]
        attn_defect_instant: 当前步计算的即时注意力掩码(未累积)
    """
    import torch.nn.functional as F
    import numpy as np
    
    # 1. 计算原始像素差异和原始缺陷注意力
    pixel_diff = torch.abs(img_free_crop - pred_crop0)
    # 在通道维度上取平均，确保形状为 (B, 1, H, W)
    if pixel_diff.shape[1] > 1: 
        pixel_diff = pixel_diff.mean(dim=1, keepdim=True)  
        
    # 归一化差异到 [0, 1]
    batch_size, _, height, width = pixel_diff.shape
    pixel_diff_flat = pixel_diff.view(batch_size, -1)
    min_vals = pixel_diff_flat.min(dim=1, keepdim=True)[0]
    max_vals = pixel_diff_flat.max(dim=1, keepdim=True)[0]
    # 防止除以零
    range_vals = max_vals - min_vals + 1e-8 
    pixel_diff_normalized = (pixel_diff_flat - min_vals) / range_vals
    pixel_diff_normalized = pixel_diff_normalized.view(batch_size, 1, height, width)

    # 使用Sigmoid函数增强对比度，得到原始注意力图
    attn_defect_raw = torch.sigmoid((pixel_diff_normalized - 0.5) / temperature)
    
    # 3. 添加中心加权 - 优先关注中心区域
    if center_weight > 0:
        # 创建中心加权掩码
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, height, device=attn_defect_raw.device),
            torch.linspace(-1, 1, width, device=attn_defect_raw.device)
        )
        # 计算到中心的距离 (0到√2)
        dist_from_center = torch.sqrt(x_grid**2 + y_grid**2)
        # 将距离映射到 [0,1] 范围
        dist_from_center = dist_from_center / torch.max(dist_from_center)
        # 创建中心权重 - 中心位置权重大，边缘位置权重小
        center_mask = torch.exp(-(dist_from_center**2) / (2 * radial_decay**2))
        # 扩展维度以匹配注意力图的形状
        center_mask = center_mask.unsqueeze(0).unsqueeze(0).expand_as(attn_defect_raw)
        
        # 应用中心权重 (center_weight控制加权程度)
        weighted_attention = attn_defect_raw * (1.0 + center_weight * center_mask)
        # 重新归一化到 [0,1]
        weighted_attention = torch.clamp(weighted_attention, 0, 1)
        attn_defect_raw = weighted_attention
    
    # 4. 对注意力图进行模糊处理，平滑边缘
    if blur_sigma > 0:
        blur_kernel_size = max(3, int(blur_sigma * 3) * 2 + 1)
        # 应用高斯模糊 (确保输入是 B, C, H, W)
        attn_defect_blurred = torch.zeros_like(attn_defect_raw)
        for b in range(attn_defect_raw.shape[0]):
             # 使用 torchvision functional 进行模糊处理
             attn_defect_blurred[b] = TF.gaussian_blur(
                 attn_defect_raw[b], # 输入是 (C, H, W)
                 kernel_size=[blur_kernel_size, blur_kernel_size],
                 sigma=[blur_sigma, blur_sigma]
             )
        attn_defect_instant = attn_defect_blurred
    else:
        attn_defect_instant = attn_defect_raw
    
    # 5. 改进的连通域分析 - 保留涉及中心区域的连通域，过滤仅在边缘的区域
    if connected_components:
        # 创建一个新的掩码来保存结果
        filtered_mask = torch.zeros_like(attn_defect_instant)
            
        # 计算中心区域的范围
        center_size = int(min(height, width) * center_region_ratio)
        center_y, center_x = height // 2, width // 2
        center_y_min = center_y - center_size // 2
        center_y_max = center_y + center_size // 2
        center_x_min = center_x - center_size // 2
        center_x_max = center_x + center_size // 2
        
        # 创建中心区域掩码
        center_region_mask = np.zeros((height, width), dtype=np.bool_)
        center_region_mask[center_y_min:center_y_max, center_x_min:center_x_max] = True
        
        # 为每个批次单独处理
        for b in range(batch_size):
            # 转换为numpy，阈值化为二值图
            mask_np = attn_defect_instant[b, 0].cpu().numpy()
            
            # 使用自适应阈值而非固定阈值
            if adaptive_threshold:
                # 提取中心区域的掩码值
                center_values = mask_np[center_region_mask]
                
                # 计算中心区域的均值作为基准
                center_mean = np.mean(center_values)
                # 使用均值乘以缩放因子作为阈值
                threshold = max(0.05, min(center_mean * threshold_factor, 0.5))  # 限制在合理范围内

                # 记录使用的阈值
                logging_info(f"使用自适应阈值: {threshold:.3f} (中心均值: {center_mean:.3f})")
            else:
                # 使用固定阈值
                threshold = 0.2
            
            # 检查是否会导致全黑蒙版
            binary_test = (mask_np > threshold).astype(np.uint8)
            if np.sum(binary_test) == 0:
                # 如果会导致全黑蒙版，寻找一个合适的阈值
                # 尝试找到能产生至少5%像素为1的最大阈值
                sorted_values = np.sort(mask_np.flatten())
                idx = max(0, int(len(sorted_values) * 0.95) - 1)  # 取前95%的最大值
                adaptive_threshold = sorted_values[idx]
                threshold = min(adaptive_threshold, 0.2)  # 不超过0.2
                logging_info(f"防止全黑蒙版: 调整阈值为 {threshold:.3f}")
            
            # 应用阈值
            binary_mask = (mask_np > threshold).astype(np.uint8)
            
            # 标记连通区域
            labeled_mask = measure.label(binary_mask, connectivity=2)
            props = measure.regionprops(labeled_mask)
            
            # 计算最小区域阈值 (基于图像大小的比例)
            min_area = height * width * min_area_ratio
            
            # 创建新的掩码，只保留足够大的区域及涉及中心的区域
            valid_mask = np.zeros_like(binary_mask, dtype=np.float32)
            for prop in props:
                # 计算该连通区域的掩码
                region_mask = labeled_mask == prop.label
                
                # 检查区域是否与中心区域有重叠
                intersects_center = np.any(region_mask & center_region_mask)
                
                # 保留条件：区域足够大且与中心区域有重叠
                if prop.area >= min_area and intersects_center:
                    valid_mask[region_mask] = mask_np[region_mask]
            
            # 如果结果仍为空，确保至少保留中心区域
            if np.sum(valid_mask) == 0:
                logging_info(f"未找到有效区域，保留中心区域的值")
                valid_mask[center_region_mask] = mask_np[center_region_mask]
            
            # 转回tensor
            filtered_mask[b, 0] = torch.from_numpy(valid_mask).to(attn_defect_instant.device)

        attn_defect_instant = filtered_mask
    
    # 6. 最终处理和返回
    attn_defect_instant = torch.clamp(attn_defect_instant, 0, 1)
    
    # 7. 如果提供了累积注意力掩码
    if attn_defect_accum is not None:
        # 使用指数移动平均(EMA)更新累积注意力
        attn_defect_final = attn_defect_accum + attn_defect_instant
        # 确保累积值在合理范围
        attn_defect_final = torch.clamp(attn_defect_final, 0, 1)
    else:
        # 如果没有提供累积掩码，则使用当前计算的即时注意力
        attn_defect_final = attn_defect_instant
        
    attn_bg_final = 1.0 - attn_defect_final # 背景注意力相应调整

    # 返回过滤后的注意力掩码以及当前步的即时注意力(用于累积)
    return attn_defect_final, attn_bg_final, attn_defect_instant


def compute_attention_masks_wrapper(pred_crop0, img_free_crop=None, threshold_bg=0.1, 
                           temperature=0.1, blur_sigma=0.5, attn_defect_accum=None,
                           center_weight=0.7, radial_decay=0.8, connected_components=True, min_area_ratio=0.01,
                           center_region_ratio=0.3, adaptive_threshold=True, threshold_factor=0.5,
                           enable_visualization=False, save_dir="", image_name="attention_process", 
                           time_step=None):
    """
    注意力蒙版计算函数的包装器，可以选择性地启用可视化功能
    
    参数:

        pred_crop0: 低分辨率缺陷模型的预测结果
        img_free_crop: 真实无缺陷背景对应的裁剪区域，用于过滤
        threshold_bg: 背景区域识别阈值，小于此值的差异被认为是可信背景
        temperature: 温度系数，用于控制注意力图的锐度
        blur_sigma: 高斯模糊的标准差，用于平滑注意力图
        attn_defect_accum: 累积的缺陷注意力掩码，如果提供则使用EMA更新
        center_weight: 中心加权的权重参数 (0-1)，越大则中心区域权重越高
        radial_decay: 径向衰减率 (0-1)，越大则权重向边缘衰减越慢
        connected_components: 是否启用连通域分析过滤
        min_area_ratio: 连通域最小面积比例，小于此值将被过滤
        center_region_ratio: 图像中心区域的大小比例 (0-1)，用于定义中心区域
        adaptive_threshold: 是否使用自适应阈值而非固定值
        threshold_factor: 自适应阈值时的缩放因子 (0-1)
        enable_visualization: 是否启用可视化功能
        save_dir: 保存可视化结果的目录
        image_name: 可视化图像的基础名称
        time_step: 当前的时间步（如果有）
        
    返回:
        attn_defect: 缺陷区域注意力掩码 [0,1]
        attn_bg: 背景区域注意力掩码 [0,1]
        attn_defect_instant: 当前步计算的即时注意力掩码(未累积)
    """
    if enable_visualization:
        # 如果时间步可用，将其添加到图像名称中
        if time_step is not None:
            image_name = f"{image_name}_t{time_step}"
            
        # 确保保存目录存在
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        logging_info(f"使用可视化功能计算注意力蒙版，结果将保存到 {save_dir}")
        
        # 调用带可视化功能的函数
        attn_defect, attn_bg, attn_defect_instant, _ = compute_attention_masks_with_vis(
            pred_crop0=pred_crop0,
            img_free_crop=img_free_crop,
            temperature=temperature,
            blur_sigma=blur_sigma,
            attn_defect_accum=attn_defect_accum,
            center_weight=center_weight,
            radial_decay=radial_decay,
            connected_components=connected_components,
            min_area_ratio=min_area_ratio,
            center_region_ratio=center_region_ratio,
            adaptive_threshold=adaptive_threshold,
            threshold_factor=threshold_factor,
            save_visualization=True,
            save_dir=save_dir,
            image_name=image_name
        )
    else:
        # 调用原始函数
        attn_defect, attn_bg, attn_defect_instant = compute_attention_masks(
            pred_crop0=pred_crop0,
            img_free_crop=img_free_crop,
            temperature=temperature,
            blur_sigma=blur_sigma,
            attn_defect_accum=attn_defect_accum,
            center_weight=center_weight,
            radial_decay=radial_decay,
            connected_components=connected_components,
            min_area_ratio=min_area_ratio,
            center_region_ratio=center_region_ratio,
            adaptive_threshold=adaptive_threshold,
            threshold_factor=threshold_factor
        )
        
    return attn_defect, attn_bg, attn_defect_instant
