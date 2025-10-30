# implement DDIM schedulers
import torch
import numpy as np


def ddim_timesteps(
    num_inference_steps,  # ddim step num
    ddpm_num_steps=1000,  # ! notice this should be 250 for celebA model
    schedule_type="linear",
    start_step=None,  # 添加起始步数参数
    **kwargs,
):
    if schedule_type == "linear":
        # linear timestep schedule
        step_ratio = ddpm_num_steps / num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int32)
        )
    elif schedule_type == "quad":
        timesteps = (
            (np.linspace(0, np.sqrt(ddpm_num_steps * 0.8), num_inference_steps)) ** 2
        ).astype(int)
    
    timesteps = timesteps.tolist()
    timesteps = sorted(list(set(timesteps)), reverse=True)  # remove duplicates
    
    # 如果指定了起始步数，则截取从该步数开始的时间步序列
    if start_step is not None:
        # 找到第一个小于等于start_step的时间步索引
        start_idx = next((i for i, step in enumerate(timesteps) if step <= start_step), 0)
        # 只保留从该索引开始的时间步
        timesteps = timesteps[start_idx:]
    
    return timesteps


def repaint_step_filter(filter_type, max_T):
    if filter_type == "none":
        return lambda x: False
    elif filter_type.startswith("firstp"):
        percent = float(filter_type.split("-")[1])
        return lambda x: x < max_T * (1.0 - percent / 100.0)  # this isreverse
    elif filter_type.startswith("lastp"):
        percent = float(filter_type.split("-")[1])
        return lambda x: x > max_T * percent / 100.0  # this isreverse
    elif filter_type.startswith("firstn"):
        num = int(filter_type.split("-")[1])
        return lambda x: x < max_T - num
    elif filter_type.startswith("lastn"):
        num = int(filter_type.split("-")[1])
        return lambda x: x > num


def ddim_repaint_timesteps(
    num_inference_steps,  # ddim step num
    ddpm_num_steps=1000,  # ! notice this should be 250 for celebA model
    jump_length=10,
    jump_n_sample=10,
    device=None,
    time_travel_filter_type="none",
    start_step=None,  # 添加起始步数参数
    **kwargs,
):
    """
    生成DDIM-Repaint采样器使用的时间步列表，支持从指定步数开始
    
    参数:
        num_inference_steps: DDIM推理步数
        ddpm_num_steps: DDPM训练时使用的总步数
        jump_length: 时间旅行的跳跃长度
        jump_n_sample: 时间旅行的重复次数
        device: 计算设备
        time_travel_filter_type: 时间旅行过滤类型
        start_step: 开始去噪的时间步 (如果为None则从最高噪声级别开始)
        
    返回:
        timesteps: 时间步列表，包含时间旅行
    """
    num_inference_steps = min(ddpm_num_steps, num_inference_steps)
    timesteps = []
    jumps = {}
    step_filter = repaint_step_filter(time_travel_filter_type, ddpm_num_steps)
    
    # 调整起始t值以支持从特定步骤开始
    if start_step is not None:
        # 将start_step转换为对应的inference step索引
        start_t = int(start_step * num_inference_steps / ddpm_num_steps)
        t_start = min(num_inference_steps, start_t)
    else:
        t_start = num_inference_steps
    
    # 设置时间旅行的跳跃点
    for j in range(0, t_start - jump_length, jump_length):
        if step_filter(j):  # don't do time travel when t is close to T
            continue
        jumps[j] = jump_n_sample - 1

    t = t_start
    while t >= 1:
        t = t - 1
        timesteps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                timesteps.append(t)
                
    # 将inference steps转换回原始模型的时间步
    timesteps = np.array(timesteps) * (ddpm_num_steps // num_inference_steps)
    timesteps = timesteps.tolist()
    
    if start_step is not None:
        print(f"从时间步 {timesteps[0]} 开始去噪 (总计 {len(timesteps)} 步，包含时间旅行)")
    
    return timesteps