import os
import numpy as np

from matplotlib.pyplot import figure

from .general_utils import get_random_time_stamp, makedir_if_not_exist

dir_figures = os.path.join(os.getcwd(), "figures")
makedir_if_not_exist(dir_figures)


def visualize_and_save_attention_process(attn_masks_dict, conf, model_kwargs, generate_gif=True):
    """
    可视化并保存整个去噪过程中的注意力掩码变化
    
    参数:
        attn_masks_dict: 字典，包含每个时间步的注意力掩码和相关图像
            格式: {
                time_step: {
                    'attn_defect': tensor,
                    'attn_bg': tensor,
                    'pred_x0_crop': tensor, 
                    'pred_crop0': tensor
                }
            }
        conf: 配置对象
        model_kwargs: 模型参数字典
        generate_gif: 布尔值，控制是否生成GIF动画，默认为True
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import matplotlib.patches as patches
    from PIL import Image
    from utils import normalize_image
    from utils.general_utils import makedir_if_not_exist
    
    if not hasattr(conf, 'attention_save_path') or not conf.attention_save_path:
        return
    
    # 确保保存目录存在
    makedir_if_not_exist(conf.attention_save_path)
    
    # 获取图像基本名称
    image_base_name = model_kwargs.get("image_name", "attention_process")
    
    # 获取原始图像
    img_full = model_kwargs.get("img", None)  # 获取完整原始图像
    img_free = model_kwargs.get("img_free", None)  # 获取无缺陷图像
    
    # 获取裁剪位置信息
    location = model_kwargs.get("location", [0, 0])  # 获取裁剪位置 [x, y]
    crop_size = conf['crop']['image_size']
    loc_x, loc_y = location
    crop_end_x, crop_end_y = loc_x + crop_size, loc_y + crop_size
    
    # 按时间步排序
    time_steps = sorted(attn_masks_dict.keys(), reverse=True)
    
    # 确定要可视化的批次数
    num_batches = attn_masks_dict[time_steps[0]]['attn_defect'].shape[0]    
    for b in range(num_batches):        
        # 创建网格图 - 7列布局：原图、背景预测、缺陷预测、缺陷注意力、即时注意力、注意力叠加、完整预测
        num_time_steps = len(time_steps)
        # 增加左边距，为行标签预留更多空间
        fig, axes = plt.subplots(num_time_steps, 7, figsize=(30, 4 * num_time_steps))
        
        # 添加全局标题
        fig.suptitle(f'Attention Mask Evolution for Sample {b+1}', fontsize=16, y=0.98)
        
        # 添加全局行标签说明，调整位置避免覆盖图像
        if num_time_steps > 1:  # 只有在有多个时间步时才添加
            fig.text(0.05, 0.5, 'Diffusion Steps', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        
        # 设置列标题
        col_titles = ['Original Image', 'Background Prediction', 'Defect Prediction', 
                     'Defect Attention', 'Instant Attention', 'Defect Overlay', 'Full Prediction']
        for i, title in enumerate(col_titles):
            if num_time_steps > 1:
                axes[0, i].set_title(title)
            else:
                axes[i].set_title(title)
        
        # 提取完整原始图像
        if img_full is not None:
            full_img_np = normalize_image(img_full[b].detach().cpu()).permute(1, 2, 0).numpy()
        else:
            # 如果没有完整图像，创建一个黑色图像
            full_img_np = np.zeros((256, 256, 3))  # 假设图像大小为256x256，根据实际情况调整
        
        # 填充每个时间步的可视化结果
        for row, t in enumerate(time_steps):
            data = attn_masks_dict[t]
              # 转换为numpy数组用于可视化
            pred_x0_crop_np = normalize_image(data['pred_x0_crop'][b].detach().cpu()).permute(1, 2, 0).numpy()
            pred_crop0_np = normalize_image(data['pred_crop0'][b].detach().cpu()).permute(1, 2, 0).numpy()
            attn_defect_np = data['attn_defect'][b, 0].detach().cpu().numpy()
            
            # 获取即时注意力掩码（如果存在）
            attn_defect_instant_np = None
            if 'attn_defect_instant' in data:
                attn_defect_instant_np = data['attn_defect_instant'][b, 0].detach().cpu().numpy()
            else:
                # 如果不存在，创建一个空白掩码
                attn_defect_instant_np = np.zeros_like(attn_defect_np)
                
            # 获取全局预测图像（如果存在）
            pred_full_np = data.get('pred_full', None)
            if pred_full_np is not None:
                pred_full_np = normalize_image(pred_full_np[b].detach().cpu()).permute(1, 2, 0).numpy()
            else:
                pred_full_np = np.copy(full_img_np)  # 如果不存在，则使用原始图像副本
            
            ax_row = axes[row] if num_time_steps > 1 else axes            # 绘制完整原始图像并标记缺陷位置
            ax_row[0].imshow(full_img_np)
            rect = patches.Rectangle((loc_x, loc_y), crop_size, crop_size, 
                                    linewidth=2, edgecolor='r', facecolor='none')
            ax_row[0].add_patch(rect)
              # 设置行标签，使用更大字体显示时间步
            ax_row[0].set_ylabel(f'Step {t}', fontsize=16)
            
            # 只关闭刻度而不关闭整个坐标轴，保留标签显示
            ax_row[0].set_xticks([])
            ax_row[0].set_yticks([])
              # 绘制背景预测
            ax_row[1].imshow(pred_x0_crop_np)
            ax_row[1].set_xticks([])
            ax_row[1].set_yticks([])
            
            # 绘制缺陷预测
            ax_row[2].imshow(pred_crop0_np)
            ax_row[2].set_xticks([])
            ax_row[2].set_yticks([])
              # 绘制缺陷注意力掩码（累积）
            im_defect = ax_row[3].imshow(attn_defect_np, cmap='hot', vmin=0, vmax=1)
            ax_row[3].set_xticks([])
            ax_row[3].set_yticks([])
            
            # 绘制即时注意力掩码（未累积）
            im_defect_instant = ax_row[4].imshow(attn_defect_instant_np, cmap='hot', vmin=0, vmax=1)
            ax_row[4].set_xticks([])
            ax_row[4].set_yticks([])
            
            # 绘制缺陷叠加效果 (Defect Overlay)
            ax_row[5].imshow(pred_x0_crop_np)
            ax_row[5].imshow(attn_defect_np, cmap='hot', alpha=0.5)
            ax_row[5].set_xticks([])
            ax_row[5].set_yticks([])
            
            # 绘制完整预测图像
            ax_row[6].imshow(pred_full_np)
            ax_row[6].set_xticks([])
            ax_row[6].set_yticks([])
          # 添加颜色条 - 累积缺陷注意力
        cbar_ax1 = fig.add_axes([0.92, 0.5, 0.01, 0.3])
        cbar1 = fig.colorbar(im_defect, cax=cbar_ax1)
        cbar1.set_label('Accumulated Attention')
        
        # 添加颜色条 - 即时缺陷注意力
        cbar_ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.3])
        cbar2 = fig.colorbar(im_defect_instant, cax=cbar_ax2)
        cbar2.set_label('Instant Attention')
        
        # 调整布局参数，为左侧标签和顶部标题预留更多空间
        plt.tight_layout(rect=[0.05, 0.03, 0.91, 0.95])
        
        # 确保目录存在，防止保存失败
        if not os.path.exists(os.path.dirname(conf.attention_save_path)):
            os.makedirs(os.path.dirname(conf.attention_save_path), exist_ok=True)
        
        # 保存图像
        process_save_path = os.path.join(conf.attention_save_path, f"{image_base_name}_attention_process_b{b+1}.png")
        plt.savefig(process_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved attention process visualization to {process_save_path}")      # 为每个批次创建动画GIF - 包含缺陷注意力掩码和预测结果比较
    if generate_gif:
        # 创建专门的gif文件夹
        gif_save_path = os.path.join(conf.attention_save_path, 'gif')
        os.makedirs(gif_save_path, exist_ok=True)
        
        for b in range(num_batches):
            # 创建缺陷注意力掩码的GIF
            gif_frames_defect = []
            # 创建对比视图GIF
            gif_frames_compare = []
            
            # 获取完整原始图像
            if img_full is not None:
                full_img_np = normalize_image(img_full[b].detach().cpu()).permute(1, 2, 0).numpy()
            else:
                full_img_np = np.zeros((256, 256, 3))  # 假设图像大小为256x256
        
        for t in time_steps:
            # 获取数据
            data = attn_masks_dict[t]
            attn_defect_np = data['attn_defect'][b, 0].detach().cpu().numpy()
            pred_x0_crop_np = normalize_image(data['pred_x0_crop'][b].detach().cpu()).permute(1, 2, 0).numpy()
            pred_crop0_np = normalize_image(data['pred_crop0'][b].detach().cpu()).permute(1, 2, 0).numpy()
            
            # 获取全局预测图像（如果存在）
            pred_full_np = data.get('pred_full', None)
            if pred_full_np is not None:
                pred_full_np = normalize_image(pred_full_np[b].detach().cpu()).permute(1, 2, 0).numpy()
            else:
                pred_full_np = np.copy(full_img_np)
              # 获取即时注意力掩码（如果存在）
            if 'attn_defect_instant' in data:
                attn_defect_instant_np = data['attn_defect_instant'][b, 0].detach().cpu().numpy()
            else:
                attn_defect_instant_np = np.zeros_like(attn_defect_np)
            
            # 将numpy数组转换为PIL图像
            frame_defect = Image.fromarray((attn_defect_np * 255).astype(np.uint8))
            frame_defect_instant = Image.fromarray((attn_defect_instant_np * 255).astype(np.uint8))
              
            # 创建对比视图
            fig, axes = plt.subplots(1, 6, figsize=(24, 4))            # 设置标题，包含当前时间步
            fig.suptitle(f'Diffusion Step {t}', fontsize=18)
              # 绘制原始图像带框
            axes[0].imshow(full_img_np)
            rect = patches.Rectangle((loc_x, loc_y), crop_size, crop_size, 
                                    linewidth=2, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].set_title('Original Image')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
              # 绘制缺陷预测
            axes[1].imshow(pred_crop0_np)
            axes[1].set_title('Defect Prediction')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
              # 绘制缺陷注意力掩码（累积）
            axes[2].imshow(attn_defect_np, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title('Accumulated Attention')
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            
            # 绘制即时注意力掩码（未累积）
            axes[3].imshow(attn_defect_instant_np, cmap='hot', vmin=0, vmax=1)
            axes[3].set_title('Instant Attention')
            axes[3].set_xticks([])
            axes[3].set_yticks([])
            
            # 绘制缺陷注意力叠加效果
            axes[4].imshow(pred_x0_crop_np)
            axes[4].imshow(attn_defect_np, cmap='hot', alpha=0.5)
            axes[4].set_title('Defect Overlay')
            axes[4].set_xticks([])
            axes[4].set_yticks([])
            
            # 绘制完整预测图像
            axes[5].imshow(pred_full_np)
            axes[5].set_title('Full Prediction')
            axes[5].set_xticks([])
            axes[5].set_yticks([])
            
            # 为GIF图提供更好的布局，确保标题有足够空间
            plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
            
            # 将图形保存到内存中的缓冲区
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frame_compare = Image.open(buf)
            plt.close()
              # 添加到帧列表
            gif_frames_defect.append(frame_defect)
            gif_frames_defect_instant = []  # 创建即时注意力掩码的GIF帧列表
            gif_frames_defect_instant.append(frame_defect_instant)
            gif_frames_compare.append(frame_compare)
          # 保存GIF
        if gif_frames_defect:
            # 确保目录存在
            os.makedirs(gif_save_path, exist_ok=True)
            
            # 保存累积缺陷注意力掩码GIF
            defect_gif_path = os.path.join(gif_save_path, f"{image_base_name}_defect_mask_evolution_b{b+1}.gif")
            gif_frames_defect[0].save(
                defect_gif_path,
                save_all=True,
                append_images=gif_frames_defect[1:],
                duration=500,  # 每帧显示500毫秒
                loop=0  # 无限循环
            )
            print(f"Saved defect attention mask evolution GIF to {defect_gif_path}")
            
            # 保存即时缺陷注意力掩码GIF
            defect_instant_gif_path = os.path.join(gif_save_path, f"{image_base_name}_instant_mask_evolution_b{b+1}.gif")
            gif_frames_defect_instant[0].save(
                defect_instant_gif_path,
                save_all=True,
                append_images=gif_frames_defect_instant[1:],
                duration=500,  # 每帧显示500毫秒
                loop=0  # 无限循环
            )
            print(f"Saved instant defect mask evolution GIF to {defect_instant_gif_path}")
            
            # 保存对比视图GIF
            compare_gif_path = os.path.join(gif_save_path, f"{image_base_name}_compare_evolution_b{b+1}.gif")
            gif_frames_compare[0].save(
                compare_gif_path,
                save_all=True,
                append_images=gif_frames_compare[1:],
                duration=1000,  # 每帧显示1秒
                loop=0
            )
            print(f"Saved comparison evolution GIF to {compare_gif_path}")





class Drawer:
    def __init__(
        self,
        num_row=1,
        num_col=1,
        unit_length=10,
        unit_row_length=None,
        unit_col_length=None,
    ):
        """
        Init the drawer with the (width=num_col*unit_length, height=num_row*unit_length).
        :param num_row: the number of rows
        :type num_row: int
        :param num_col: the number of columns
        :type num_col: int
        :param unit_length: the length of unit
        :type unit_length: float
        :param unit_row_length: the length of unit for rows
        :param unit_col_length: the length of unit for cols
        """
        self.num_row = num_row
        self.num_col = num_col
        unit_row_length = unit_length if unit_row_length is None else unit_row_length
        unit_col_length = unit_length if unit_col_length is None else unit_col_length
        self.figure = figure(
            figsize=(num_col * unit_row_length, num_row * unit_col_length)
        )

    def add_one_empty_axes(
        self,
        index=1,
        nrows=None,
        ncols=None,
        title="",
        xlabel="",
        ylabel="",
        fontsize=15,
        xlim=None,
        ylim=None,
    ):
        """
        Draw one axes, which can be understood as a sub-figure.
        :param index: The subplot will take the index position on a grid with nrows rows and ncols columns.
        :type index: int
        :param nrows: the number of rows in the figure
        :type nrows: int
        :param ncols: the number of columns in the figure
        :type ncols: int
        :param title: the title of the axes
        :type title: str
        :param xlabel: the label for x axis
        :type xlabel: str
        :param ylabel: the label for x axis
        :type ylabel: str
        :param fontsize: the size of the fonts
        :param xlim: the range of x axis, (low, upp)
        :param ylim: the range of y axis, (low, upp)
        :return:
        :rtype:
        """
        nrows = self.num_row if nrows is None else nrows
        ncols = self.num_col if ncols is None else ncols

        ax = self.figure.add_subplot(nrows, ncols, index)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        return ax

    def draw_one_axes(
        self,
        x,
        y,
        labels=None,
        *,
        index=1,
        nrows=None,
        ncols=None,
        title="",
        xlabel="",
        ylabel="",
        use_marker=False,
        linewidth=6,
        fontsize=15,
        xlim=None,
        ylim=None,
        smooth=0,
        mode="plot",
        **kwargs
    ):
        """
        Draw one axes, which can be understood as a sub-figure.
        :param x: the data for x axis, list
        :param y: the data for y axis, list of line lists. e.g. [[1, 2, 3], [2, 3, 1]], list
        :param labels: the list of labels of each line, list
        :param index: The subplot will take the index position on a grid with nrows rows and ncols columns.
        :type index: int
        :param nrows: the number of rows in the figure
        :type nrows: int
        :param ncols: the number of columns in the figure
        :type ncols: int
        :param title: the title of the axes
        :type title: str
        :param xlabel: the label for x axis
        :type xlabel: str
        :param ylabel: the label for x axis
        :type ylabel: str
        :param use_marker: whether use markers to mark the points, default=False
        :type use_marker: bool
        :param linewidth: the width of the lines for mode "plot", or the size of the points for mode "scatter"
        :param fontsize: the size of the fonts
        :param xlim: the range of x axis, (low, upp)
        :param ylim: the range of y axis, (low, upp)
        :param smooth: smooth the line with neighbours
        :param mode: "plot" or "scatter"
        :return:
        :rtype:
        """
        ax = self.add_one_empty_axes(
            index, nrows, ncols, title, xlabel, ylabel, fontsize, xlim, ylim
        )

        format_generator = self.get_format(use_marker)
        for i, yi in enumerate(y):
            if len(x) == len(y) and type(x[0]) is list:
                xi = x[i]
            elif len(x) == len(y[0]) and type(x[0]) is not list:
                xi = x
            else:
                raise NotImplementedError

            if smooth != 0:
                yi_smoothed = []
                for j, yij in enumerate(yi):
                    _r = min(j + smooth, len(yi) - 1)
                    _l = max(j - smooth, 0)
                    yij = sum(yi[_l:_r]) / (_r - _l)
                    yi_smoothed.append(yij)
                yi = yi_smoothed

            len_no_nan = 0
            while len_no_nan < len(yi) and not (
                np.isnan(yi[len_no_nan]) or np.isinf(yi[len_no_nan])
            ):
                len_no_nan += 1
            if len_no_nan == 0:
                continue

            fmt = next(format_generator)

            if labels is not None:
                kwargs["label"] = labels[i]
            if mode == "plot":
                kwargs["linewidth"] = linewidth

            if mode == "plot":
                ax.plot(xi[:len_no_nan], yi[:len_no_nan], fmt, **kwargs)
            elif mode == "scatter":
                ax.scatter(
                    xi[:len_no_nan], yi[:len_no_nan], c=fmt[0], s=linewidth, **kwargs
                )
            else:
                raise NotImplementedError

        if labels is not None:
            ax.legend(fontsize=fontsize)

        return ax

    def show(self):
        """
        To show the figure.
        """
        self.figure.show()

    def save(self, fname=None):
        """
        To save the figure as fname.
        :param fname: the filename
        :type fname: str
        """
        if fname is None:
            fname = get_random_time_stamp()
        fname = "%s.jpeg" % fname if not fname.endswith(".config") else fname
        self.figure.savefig(os.path.join(dir_figures, fname), bbox_inches="tight")

    def clear(self):
        """
        Clear the figure.
        """
        self.figure.clf()

    @staticmethod
    def get_format(use_marker=False):
        """
        Get the format of a line.
        :param use_marker: whether use markers for points or not.
        :type use_marker: bool
        """
        p_color, p_style, p_marker = 0, 0, 0
        colors = ["r", "g", "b", "c", "m", "y", "k"]
        styles = ["-", "--", "-.", ":"]
        markers = [""]
        if use_marker:
            markers = [
                "o",
                "v",
                "^",
                "<",
                ">",
                "1",
                "2",
                "3",
                "4",
                "8",
                "s",
                "p",
                "P",
                "*",
                "h",
                "H",
                "+",
                "x",
                "X",
                "D",
                "d",
                "|",
                "_",
            ]

        while True:
            yield colors[p_color] + styles[p_style] + markers[p_marker]
            p_color += 1
            p_style += 1
            p_marker += 1
            p_color %= len(colors)
            p_style %= len(styles)
            p_marker %= len(markers)
