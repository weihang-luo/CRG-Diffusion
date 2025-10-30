import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
from torchvision.transforms import functional as TF
from utils.logger import logging_info
from skimage import measure

# 设置中文字体支持
def set_chinese_font():
    """设置matplotlib的中文字体支持"""
    import matplotlib.font_manager as fm
    from pathlib import Path
    import sys
    
    # 设置matplotlib基本配置，确保可以显示负号
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取当前系统中所有可用字体
    system_fonts = fm.findSystemFonts()
    logging_info(f"系统中找到 {len(system_fonts)} 个字体")
    
    # 尝试从系统字体中查找中文字体
    chinese_fonts = []
    for font_path in system_fonts:
        try:
            font = fm.FontProperties(fname=font_path)
            font_name = font.get_name()
            # 查找可能的中文字体(包含这些关键词的可能是中文字体)
            if any(keyword in font_name.lower() for keyword in 
                   ['simsun', 'simhei', 'kaiti', 'heiti', 'microsoftyahei', 'microhei', 
                    'pingfang', 'noto sans cjk', 'noto serif cjk', 'source han']):
                chinese_fonts.append((font_name, font_path))
                logging_info(f"找到可能的中文字体: {font_name} 路径: {font_path}")
        except:
            continue
    
    # 如果找到了可能的中文字体，使用第一个
    if chinese_fonts:
        font_name, font_path = chinese_fonts[0]
        font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        logging_info(f"使用中文字体: {font_name}")
        return font
    
    # 如果没有找到中文字体，尝试系统中常见的中文字体
    font_candidates = [
        'SimHei', 'SimSun', 'Heiti TC', 'Heiti SC', 'AR PL UKai CN', 'AR PL UMing CN', 
        'WenQuanYi Micro Hei', 'Microsoft YaHei', 'PingFang SC', 'STFangsong',
        'FangSong', 'STSong', 'STXihei', 'KaiTi', 'Noto Sans CJK SC', 'Source Han Sans CN'
    ]
    
    for font_name in font_candidates:
        try:
            font = fm.FontProperties(family=font_name)
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            logging_info(f"使用备选中文字体: {font_name}")
            return font
        except:
            continue
    
    # 如果仍然没找到合适的字体，使用默认设置，并采用编码方式显示中文
    logging_info("未找到可用的中文字体，将使用Unicode编码显示中文字符")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    return None

# 阶段描述字典 - 为每个阶段提供详细说明
STAGE_DESCRIPTIONS = {
    "1_原始差异": "计算低分辨率缺陷模型与无缺陷图像的像素差异，生成初始注意力图",
    "2_背景过滤": "使用高分辨率背景模型的预测过滤掉低置信度区域，减少假阳性",
    "3_中心加权": "优先关注图像中心区域，使用径向权重增强中心区域的注意力信号",
    "4_高斯模糊": "对注意力图应用高斯模糊，平滑边缘并减少噪声",
    "5_连通域分析": "过滤小的、与中心无关的连通区域，保留主要缺陷区域",
    "6_累积更新": "使用指数移动平均(EMA)更新累积注意力，提高时间一致性",
    "6_最终结果": "生成最终的缺陷注意力掩码，用于后续处理"
}

# 英文备用描述 - 当中文无法显示时使用
STAGE_DESCRIPTIONS_EN = {
    "1_原始差异": "Original Pixel Difference",
    "2_背景过滤": "Background Filtering",
    "3_中心加权": "Center Weighting",
    "4_高斯模糊": "Gaussian Blur",
    "5_连通域分析": "Connected Components Analysis",
    "6_累积更新": "Accumulated Update",
    "6_最终结果": "Final Result"
}

# 掩码名称字典 - 为每个掩码类型提供说明
MASK_DESCRIPTIONS = {
    "pixel_diff_normalized": "归一化的像素差异",
    "attn_defect_raw": "原始缺陷注意力",
    "diff_bg_ref": "背景模型与真实背景的差异",
    "mask_bg_conf": "背景可信度掩码",
    "attn_defect_filtered": "过滤后的缺陷注意力",
    "attn_defect_to_blur": "待模糊处理的注意力",
    "center_mask": "中心加权掩码",
    "weighted_attention": "加权后的注意力",
    "attn_defect_blurred": "模糊处理后的注意力",
    "attn_defect_instant": "当前步的即时注意力",
    "attn_defect_accum_before": "累积前的注意力",
    "attn_defect_final": "最终缺陷注意力掩码"
}

# 英文备用描述 - 当中文无法显示时使用
MASK_DESCRIPTIONS_EN = {
    "pixel_diff_normalized": "Normalized Pixel Difference",
    "attn_defect_raw": "Raw Defect Attention",
    "diff_bg_ref": "Background Model Difference",
    "mask_bg_conf": "Background Confidence Mask",
    "attn_defect_filtered": "Filtered Defect Attention",
    "attn_defect_to_blur": "Attention Before Blur",
    "center_mask": "Center Weighting Mask",
    "weighted_attention": "Weighted Attention",
    "attn_defect_blurred": "Blurred Attention",
    "attn_defect_instant": "Current Step Attention",
    "attn_defect_accum_before": "Attention Before Accumulation",
    "attn_defect_final": "Final Defect Attention Mask"
}

# 安全地获取中文描述，如果中文不可用则回退到英文
def safe_get_description(key, descriptions_dict, descriptions_en_dict):
    """安全地获取描述文本，如果中文描述不能正常显示则使用英文"""
    cn_text = descriptions_dict.get(key, key)
    en_text = descriptions_en_dict.get(key, key)
    return cn_text, en_text

def visualize_masks(masks_dict, save_dir, image_name="attention_process", batch_idx=0):
    """
    可视化注意力蒙版的生成过程
    
    参数:
        masks_dict: 包含各阶段注意力蒙版的字典
        save_dir: 保存可视化结果的目录
        image_name: 图像基础名称
        batch_idx: 批次索引
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    chinese_font = set_chinese_font()
    
    # 按步骤顺序获取阶段名称
    stage_names = list(masks_dict.keys())
    
    # 确定每个阶段的掩码数量，并计算最大值来确定列数
    max_masks_per_stage = max([len(masks_dict[stage]) for stage in stage_names])
    
    # 创建图形和子图布局
    fig = plt.figure(figsize=(max(16, max_masks_per_stage * 4), len(stage_names) * 4))
    
    # 创建GridSpec对象以管理子图布局
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(len(stage_names), max_masks_per_stage + 1)  # 额外一列用于显示阶段描述
    
    # 设置总标题 - 中英文双语标题
    if chinese_font:
        fig.suptitle(f'注意力蒙版生成过程 (批次 {batch_idx+1})', fontsize=16, fontproperties=chinese_font)
    else:
        fig.suptitle(f'Attention Mask Generation Process (Batch {batch_idx+1})', fontsize=16)
    
    # 遍历所有阶段并可视化
    for i, stage in enumerate(stage_names):
        data = masks_dict[stage]
        stage_name = stage.split('_', 1)[1] if '_' in stage else stage  # 去掉序号前缀
        
        # 获取阶段描述（中英文）
        stage_cn_desc, stage_en_desc = safe_get_description(stage, STAGE_DESCRIPTIONS, STAGE_DESCRIPTIONS_EN)
        
        # 添加阶段描述文本框
        ax_desc = fig.add_subplot(gs[i, 0])
        ax_desc.axis('off')
        
        # 添加步骤标题 - 使用中文或英文
        if chinese_font:
            ax_desc.text(0.05, 0.5, f'步骤 {i+1}:\n{stage_name}', 
                        fontsize=12, fontproperties=chinese_font,
                        verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightskyblue', alpha=0.3))
        else:
            ax_desc.text(0.05, 0.5, f'Step {i+1}:\n{stage_en_desc}', 
                        fontsize=12,
                        verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightskyblue', alpha=0.3))
        
        # 添加阶段描述 - 使用中文或英文
        if chinese_font:
            ax_desc.text(0.05, 0.2, stage_cn_desc, 
                        fontsize=10, fontproperties=chinese_font,
                        verticalalignment='center', wrap=True,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        else:
            ax_desc.text(0.05, 0.2, stage_en_desc, 
                        fontsize=10,
                        verticalalignment='center', wrap=True,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 获取当前阶段的所有蒙版并按名称排序显示
        sorted_items = sorted(data.items())
        
        # 遍历当前阶段所有掩码
        for j, (mask_name, mask) in enumerate(sorted_items):
            # 创建子图
            ax = fig.add_subplot(gs[i, j+1])  # 加1是因为第一列用于阶段描述
            
            # 转换为numpy用于可视化
            if isinstance(mask, torch.Tensor):
                mask_np = mask[batch_idx, 0].detach().cpu().numpy()
            else:
                mask_np = mask
            
            # 绘制蒙版
            im = ax.imshow(mask_np, cmap='hot', vmin=0, vmax=1)
            
            # 获取掩码描述（中英文）
            mask_cn_desc, mask_en_desc = safe_get_description(mask_name, MASK_DESCRIPTIONS, MASK_DESCRIPTIONS_EN)
            
            # 添加掩码名称标题 - 使用中文或英文
            if chinese_font:
                ax.set_title(mask_cn_desc, fontsize=10, fontproperties=chinese_font)
            else:
                ax.set_title(mask_en_desc, fontsize=10)
                
            ax.axis('off')
            
            # 添加颜色条（只在每行最后一个掩码添加）
            if j == len(sorted_items) - 1:
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
                if chinese_font:
                    cbar.set_label('注意力强度', fontproperties=chinese_font)
                else:
                    cbar.set_label('Attention Intensity')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题留出空间
    
    # 保存图像
    save_path = os.path.join(save_dir, f"{image_name}_mask_stages_b{batch_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging_info(f"已保存注意力蒙版生成过程可视化结果到 {save_path}")
    
    # 创建更高质量的GIF动画来显示蒙版演变
    gif_frames = []
    
    for i, stage in enumerate(stage_names):
        data = masks_dict[stage]
        # 使用最后一个蒙版(通常是该阶段的最终结果)
        mask_name = list(sorted(data.keys()))[-1]
        mask = data[mask_name]
        
        if isinstance(mask, torch.Tensor):
            mask_np = mask[batch_idx, 0].detach().cpu().numpy()
        else:
            mask_np = mask
        
        # 创建带标题的图像
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(mask_np, cmap='hot', vmin=0, vmax=1)
        
        # 获取不带序号的阶段名
        stage_name = stage.split('_', 1)[1] if '_' in stage else stage
        
        # 获取中英文描述
        stage_cn_desc, stage_en_desc = safe_get_description(stage, STAGE_DESCRIPTIONS, STAGE_DESCRIPTIONS_EN)
        mask_cn_desc, mask_en_desc = safe_get_description(mask_name, MASK_DESCRIPTIONS, MASK_DESCRIPTIONS_EN)
        
        # 设置有意义的标题 - 使用中文或英文
        if chinese_font:
            title = f'步骤 {i+1}: {stage_name}\n{mask_cn_desc}'
            ax.set_title(title, fontsize=12, fontproperties=chinese_font)
        else:
            title = f'Step {i+1}: {stage_en_desc}\n{mask_en_desc}'
            ax.set_title(title, fontsize=12)
        
        # 添加阶段描述为注释 - 使用中文或英文
        if chinese_font:
            ax.annotate(stage_cn_desc,
                       xy=(0.5, 0.02),
                       xycoords='figure fraction',
                       horizontalalignment='center',
                       fontsize=10,
                       fontproperties=chinese_font,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        else:
            ax.annotate(stage_en_desc,
                       xy=(0.5, 0.02),
                       xycoords='figure fraction',
                       horizontalalignment='center',
                       fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        ax.axis('off')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        if chinese_font:
            cbar.set_label('注意力强度', fontproperties=chinese_font)
        else:
            cbar.set_label('Attention Intensity')
        
        plt.tight_layout()
        
        # 将图像保存到内存缓冲区
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frame = Image.open(buf)
        gif_frames.append(frame)
        plt.close()
    
    # 保存GIF
    if gif_frames:
        gif_path = os.path.join(save_dir, f"{image_name}_mask_evolution_b{batch_idx}.gif")
        # 保存GIF动画，每帧持续1.5秒
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], 
                         duration=1500, loop=0)
        logging_info(f"已保存注意力蒙版演变GIF动画到 {gif_path}")

def compute_attention_masks_with_vis(pred_crop0, img_free_crop=None, 
                               temperature=0.1, blur_sigma=0.5, attn_defect_accum=None,
                               center_weight=0.7, radial_decay=0.8, connected_components=True, min_area_ratio=0.01,
                               center_region_ratio=0.3, adaptive_threshold=True, threshold_factor=0.5,
                               save_visualization=False, save_dir="", image_name="attention_process"):
    """
    计算注意力掩码，基于像素差异，并可视化整个生成过程
    
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
        save_visualization: 是否保存可视化结果
        save_dir: 保存可视化结果的目录
        image_name: 图像基础名称
        
    返回:
        attn_defect_final: 缺陷区域注意力掩码 [0,1]
        attn_bg_final: 背景区域注意力掩码 [0,1]
        attn_defect_instant: 当前步计算的即时注意力掩码(未累积)
    """
    
    # 用于保存各阶段的注意力蒙版
    mask_stages_dict = {}
    
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
    
    # 保存第一阶段结果
    mask_stages_dict["1_原始差异"] = {
        "pixel_diff_normalized": pixel_diff_normalized,
        "attn_defect_raw": attn_defect_raw
    }

    attn_defect_to_blur = attn_defect_raw
    mask_stages_dict["2_背景过滤"] = {
        "attn_defect_to_blur": attn_defect_to_blur
    }
    
    # 3. 添加中心加权 - 优先关注中心区域
    weighted_attention = attn_defect_to_blur.clone()
    if center_weight > 0:
        # 创建中心加权掩码
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, height, device=attn_defect_to_blur.device),
            torch.linspace(-1, 1, width, device=attn_defect_to_blur.device)
        )
        # 计算到中心的距离 (0到√2)
        dist_from_center = torch.sqrt(x_grid**2 + y_grid**2)
        # 将距离映射到 [0,1] 范围
        dist_from_center = dist_from_center / torch.max(dist_from_center)
        # 创建中心权重 - 中心位置权重大，边缘位置权重小
        center_mask = torch.exp(-(dist_from_center**2) / (2 * radial_decay**2))
        # 扩展维度以匹配注意力图的形状
        center_mask = center_mask.unsqueeze(0).unsqueeze(0).expand_as(attn_defect_to_blur)
        
        # 应用中心权重 (center_weight控制加权程度)
        weighted_attention = attn_defect_to_blur * (1.0 + center_weight * center_mask)
        # 重新归一化到 [0,1]
        weighted_attention = torch.clamp(weighted_attention, 0, 1)
        attn_defect_to_blur = weighted_attention
        
        # 保存中心加权结果
        mask_stages_dict["3_中心加权"] = {
            "center_mask": center_mask,
            "weighted_attention": weighted_attention
        }
    else:
        mask_stages_dict["3_中心加权"] = {
            "weighted_attention": weighted_attention
        }
    
    # 4. 对注意力图进行模糊处理，平滑边缘
    if blur_sigma > 0:
        blur_kernel_size = max(3, int(blur_sigma * 3) * 2 + 1)
        # 应用高斯模糊 (确保输入是 B, C, H, W)
        attn_defect_blurred = torch.zeros_like(attn_defect_to_blur)
        for b in range(attn_defect_to_blur.shape[0]):
             # 使用 torchvision functional 进行模糊处理
             attn_defect_blurred[b] = TF.gaussian_blur(
                 attn_defect_to_blur[b], # 输入是 (C, H, W)
                 kernel_size=[blur_kernel_size, blur_kernel_size],
                 sigma=[blur_sigma, blur_sigma]
             )
        attn_defect_instant = attn_defect_blurred
        
        # 保存模糊处理结果
        mask_stages_dict["4_高斯模糊"] = {
            "attn_defect_blurred": attn_defect_blurred
        }
    else:
        attn_defect_instant = attn_defect_to_blur
        mask_stages_dict["4_高斯模糊"] = {
            "attn_defect_instant": attn_defect_instant
        }
    
    # 5. 改进的连通域分析 - 保留涉及中心区域的连通域，过滤仅在边缘的区域
    filtered_mask = torch.zeros_like(attn_defect_instant)
    
    if connected_components:
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
                
                if len(center_values) > 0:
                    # 计算中心区域的均值作为基准
                    center_mean = np.mean(center_values)
                    # 使用均值乘以缩放因子作为阈值
                    threshold = max(0.05, min(center_mean * threshold_factor, 0.5))  # 限制在合理范围内
                else:
                    # 回退到默认值
                    threshold = 0.2
                    
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
    else:
        filtered_mask = attn_defect_instant
    
    # 保存连通域分析结果
    mask_stages_dict["5_连通域分析"] = {
        "attn_defect_filtered": filtered_mask
    }
    
    # 6. 最终处理和返回
    attn_defect_instant = torch.clamp(attn_defect_instant, 0, 1)
    
    # 7. 如果提供了累积注意力掩码，则使用EMA更新
    if attn_defect_accum is not None:
        # 使用指数移动平均(EMA)更新累积注意力
        attn_defect_final = attn_defect_accum + attn_defect_instant
        # 确保累积值在合理范围
        attn_defect_final = torch.clamp(attn_defect_final, 0, 1)
        
        # 保存累积后的结果
        mask_stages_dict["6_累积更新"] = {
            "attn_defect_accum_before": attn_defect_accum,
            "attn_defect_instant": attn_defect_instant,
            "attn_defect_final": attn_defect_final
        }
    else:
        # 如果没有提供累积掩码，则使用当前计算的即时注意力
        attn_defect_final = attn_defect_instant
        
        # 保存最终结果
        mask_stages_dict["6_最终结果"] = {
            "attn_defect_final": attn_defect_final
        }
        
    attn_bg_final = 1.0 - attn_defect_final # 背景注意力相应调整
    
    # 可视化和保存过程
    if save_visualization and save_dir:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 可视化各阶段的蒙版生成过程
        for b in range(batch_size):
            visualize_masks(mask_stages_dict, save_dir, image_name, b)

    # 返回过滤后的注意力掩码以及当前步的即时注意力(用于累积)
    return attn_defect_final, attn_bg_final, attn_defect_instant, mask_stages_dict if save_visualization else None
