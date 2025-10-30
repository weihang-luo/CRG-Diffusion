import json
import os
from pathlib import Path

def load_labels():
    """加载标签文件，返回标签名到索引的映射"""
    labels_file = Path("data/sample/labels.txt")
    
    if not labels_file.exists():
        print(f"错误：找不到标签文件 {labels_file}")
        return None
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    
    # 创建标签名到索引的映射
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    print(f"加载的标签映射: {label_to_index}")
    return label_to_index

def convert_json_to_yolo(json_file_path, label_to_index):
    """将单个JSON文件转换为YOLO格式的TXT文件"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        img_width = data.get('imageWidth', 256)
        img_height = data.get('imageHeight', 256)
        
        # 准备YOLO格式的标注行
        yolo_lines = []
        
        # 处理每个标注
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            shape_type = shape.get('shape_type', '')
            points = shape.get('points', [])
            
            # 检查标签是否在映射中
            if label not in label_to_index:
                print(f"警告：未知标签 '{label}' 在文件 {json_file_path}")
                continue
            
            # 获取类别索引
            class_id = label_to_index[label]
            
            # 处理点标注
            if shape_type == 'point' and points:
                # 取第一个点（对于点标注通常只有一个点）
                if len(points) > 0 and len(points[0]) >= 2:
                    x, y = points[0][0], points[0][1]
                    
                    # 转换为相对坐标
                    x_rel = x / img_width
                    y_rel = y / img_height
                    
                    # 确保坐标在[0,1]范围内
                    x_rel = max(0, min(1, x_rel))
                    y_rel = max(0, min(1, y_rel))
                    
                    # 格式化为YOLO格式：class_id x_center y_center
                    yolo_line = f"{class_id} {x_rel:.6f} {y_rel:.6f}"
                    yolo_lines.append(yolo_line)
        
        # 生成输出文件路径（与JSON文件同名但扩展名为.txt）
        txt_file_path = json_file_path.with_suffix('.txt')
        
        # 写入YOLO格式文件
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
            if yolo_lines:  # 如果有内容，添加最后的换行符
                f.write('\n')
        
        print(f"已转换: {json_file_path.name} -> {txt_file_path.name} ({len(yolo_lines)} 个标注)")
        return True
        
    except Exception as e:
        print(f"转换文件 {json_file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    # 加载标签映射
    label_to_index = load_labels()
    if label_to_index is None:
        return
    
    # 设置sampled_images_300文件夹路径
    sampled_dir = Path("data/sample")
    
    if not sampled_dir.exists():
        print(f"错误：找不到目录 {sampled_dir}")
        return
    
    # 查找所有JSON文件
    json_files = list(sampled_dir.glob("*.json"))
    
    if not json_files:
        print(f"错误：在 {sampled_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 转换统计
    successful_conversions = 0
    failed_conversions = 0
    
    # 逐个转换JSON文件
    for json_file in json_files:
        if convert_json_to_yolo(json_file, label_to_index):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    # 输出转换结果统计
    print(f"\n转换完成:")
    print(f"成功转换: {successful_conversions} 个文件")
    print(f"转换失败: {failed_conversions} 个文件")
    
    # 显示标签统计信息
    print(f"\n标签映射:")
    for label, idx in label_to_index.items():
        print(f"  {idx}: {label}")

if __name__ == "__main__":
    main()
