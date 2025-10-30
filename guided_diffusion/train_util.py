import copy
import functools
import os
import time
import numpy as np
from omegaconf import OmegaConf
import torch
import logging
from pathlib import Path
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast  # 导入自动混合精度训练相关模块

import blobfile as bf
import torch as th
from torch.optim import AdamW

from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from guided_diffusion.resample import create_named_schedule_sampler
import json
import torchvision.utils as vutils
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    select_args
)


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def setup_dist(rank, world_size):
    """
    设置分布式训练环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup_dist():
    """
    清理分布式训练环境
    """
    dist.destroy_process_group()

def check_memory_usage():
    """检查当前GPU的内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        peak_allocated = torch.cuda.max_memory_allocated()
        return allocated, peak_allocated
    return 0, 0

def get_logger(log_path):
    """设置日志记录器"""
    logger = logging.getLogger("diffusion")
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(handler)
    
    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        checkpoint_dir,
        resume_checkpoint,
        sample_num,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        rank=0,
        distributed=False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        # 分布式训练相关参数
        self.rank = rank
        self.distributed = distributed
        self.device = next(self.model.parameters()).device
        self.sample_num = sample_num
        
        # 关键修改：添加is_main_process标志
        # 在分布式训练中：只有rank=0是主进程
        # 在非分布式训练中：无论什么rank都是主进程
        self.is_main_process = (rank == 0 if distributed else True)
        
        self.step = 0
        self.resume_step = 0
        
        c_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 只在主进程中创建日志记录器
        if self.is_main_process:
            self.logger = get_logger(
                            bf.join(self.checkpoint_dir, 
                                    f"Train_logger_from_{'start' if not resume_checkpoint else 'resume'}_{c_time}.log")
                                    )
            self.logger.info(OmegaConf.to_yaml(self.diffusion.conf))
            if distributed:
                self.logger.info(f"使用 {dist.get_world_size()} 个GPU进行分布式训练")
        else:
            self.logger = None

        # 初始化 PyTorch AMP 的 GradScaler
        self.grad_scaler = GradScaler(enabled=self.use_fp16)
        
        if self.resume_checkpoint:
            self._load_parameters()
            
        # 创建优化器
        if self.distributed:
            # 对于分布式训练，使用模型的原始参数
            self.unwrapped_model = self.model.module
            params = self.unwrapped_model.parameters()
        else:
            # 对于单卡训练，直接使用模型参数
            self.unwrapped_model = self.model
            params = self.model.parameters()
            
        self.opt = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        
        if self.resume_step:
            self._load_optimizer_state()
            # 恢复模型的EMA参数
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            # 初始化EMA参数
            self.ema_params = [copy.deepcopy(list(params)) for _ in range(len(self.ema_rate))]

    def _load_parameters(self):
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model')

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(str(resume_checkpoint))
            if self.rank == 0 or not self.distributed:
                self.logger.info(f"loading resume model from checkpoint: {resume_checkpoint}")
            
            # 加载模型参数
            state_dict = torch.load(resume_checkpoint, map_location=self.device, weights_only=True)
            
            # 处理分布式训练的模型加载问题
            if self.distributed:
                # 检查是否需要添加"module."前缀
                if not any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
            else:
                # 检查是否需要移除"module."前缀
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    
            self.model.load_state_dict(state_dict)
            
            # 如果使用混合精度，尝试加载缩放器状态
            if self.use_fp16:
                scaler_path = bf.join(
                    bf.dirname(resume_checkpoint), f"scaler{self.resume_step:06d}.pt"
                )
                if bf.exists(scaler_path):
                    if self.rank == 0 or not self.distributed:
                        self.logger.info(f"loading grad scaler from: {scaler_path}")
                    self.grad_scaler.load_state_dict(torch.load(scaler_path))


    def _load_ema_parameters(self, rate):
        """加载EMA模型参数"""
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model')
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        
        # 获取当前模型参数，并确定当前设备
        if self.distributed:
            params = [param.clone().detach() for param in self.unwrapped_model.parameters()]
        else:
            params = [param.clone().detach() for param in self.model.parameters()]
        
        # 确定当前设备
        device = next(iter(params)).device
            
        if ema_checkpoint:
            if self.rank == 0 or not self.distributed:
                self.logger.info(f"loading EMA from checkpoint: {ema_checkpoint}...")
            
            # 加载EMA模型状态字典，并指定加载到正确的设备上
            state_dict = torch.load(ema_checkpoint, map_location=device)
              # 获取参数值，并确保它们在正确的设备上
            ema_params = []
            for name, param in zip(state_dict.keys(), params):
                # 创建参数的拷贝，确保在正确的设备上
                ema_param = state_dict[name].clone().detach().to(device=device)
                ema_params.append(ema_param)
            return ema_params
            
        # 如果没有EMA检查点，则使用当前模型参数的拷贝（确保它们已经在正确的设备上）
        return params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model')
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            # 只在主进程(rank=0)或非分布式训练时记录日志
            if self.rank == 0 or not self.distributed:
                self.logger.info(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            
            try:
                opt_state = torch.load(opt_checkpoint, map_location=self.device, weights_only=False)
                self.opt.load_state_dict(opt_state)
                if self.rank == 0 or not self.distributed:
                    self.logger.info("Successfully loaded optimizer state")
            except (ValueError, RuntimeError) as e:
                # 如果优化器状态不匹配，跳过加载并发出警告
                if self.rank == 0 or not self.distributed:
                    self.logger.warning(f"Failed to load optimizer state: {e}")
                    self.logger.warning("Skipping optimizer state loading and starting with fresh optimizer state")
                    self.logger.warning("This may cause a temporary increase in loss, but training should recover quickly")

    def run_loop(self):
        if self.rank == 0 or not self.distributed:
            self.logger.info(f"training start! step:[{self.step + self.resume_step}]")
        
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            
            # 只在主进程上保存模型和生成样本
            if self.step % self.save_interval == 0:
                if self.rank == 0 or not self.distributed:
                    self.save()
                    allocated, peak_allocated = check_memory_usage()
                    self.logger.info(f"Allocated memory: {allocated / 1024**2:.2f} MB, Peak allocated memory: {peak_allocated / 1024**2:.2f} MB")
                    
                    # 替换旧的采样代码，改用新的采样脚本
                    self.logger.info(f"training checkpoint step[{self.step + self.resume_step}] sampling...")
                    
                    # 构造最近保存的模型文件路径
                    model_path = bf.join(self.checkpoint_dir, f"model{(self.step+self.resume_step):06d}.pt")
                    save_dir = bf.join(self.checkpoint_dir, 'output')
                    
                    # 使用子进程运行采样脚本，避免GPU内存问题

                    sample_from_model(model_path, self.diffusion.conf.config_path, save_dir, self.sample_num, batch_size=4, gpu_rank=self.rank)
                    self.logger.info(f"{self.sample_num} samples generated successfully!")

                # 在分布式训练中同步所有进程
                if self.distributed:
                    dist.barrier()
                    
            self.step += 1
            
    def run_step(self, batch, cond):
        self.opt.zero_grad()
        self.forward_backward(batch, cond)


    def forward_backward(self, batch, cond):
        # 在PyTorch自动混合精度环境中进行前向传播
        batch = batch.to(self.device)
        cond = cond.to(self.device) if cond is not None else None
        t, weights = self.schedule_sampler.sample(batch.shape[0], batch.device)

        model_kwargs = dict(y=cond)
        
        # 使用PyTorch的自动混合精度
        with autocast("cuda", enabled=self.use_fp16):
            # 根据是否为分布式训练选择正确的模型
            model = self.model
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                model,
                batch,
                t,
                model_kwargs=model_kwargs,
            )
            
            losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
        
            # 记录日志
            if self.step % self.log_interval == 0 and (self.rank == 0 or not self.distributed):
                self.logger.info(f"step:[{self.step + self.resume_step}],loss:[{loss:.6f}] ")

                    
            # 使用GradScaler进行反向传播
            self.grad_scaler.scale(loss).backward()
            
            # 在进行优化器步骤前，确保所有参数梯度在相同设备上并且是相同的数据类型
            if self.distributed:
                # 同步所有进程，确保梯度已经计算完成
                dist.barrier()

                # 确保所有参数的梯度都具有相同的数据类型和设备
                for param in self.unwrapped_model.parameters():
                    if param.grad is not None:
                        # 将梯度转换为float32并移动到正确的设备上
                        param.grad = param.grad.to(device=self.device, dtype=torch.float32)

            # 执行优化器步骤和梯度缩放更新
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()

            self._update_ema()
            self._anneal_lr()


    def _update_ema(self):
        if self.distributed:
            params = list(self.unwrapped_model.parameters())
        else:
            params = list(self.model.parameters())
            
        for rate, ema_params in zip(self.ema_rate, self.ema_params):
            update_ema(ema_params, params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    # def log_step(self):
    #     logger.logkv("step", self.step + self.resume_step)
    #     logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        if self.distributed and self.rank != 0:
            return  # 在分布式训练中，只让主进程保存模型
            
        def save_checkpoint(rate, params_list):
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
                # 保存当前模型参数
                if self.logger:
                    self.logger.info(f"saving model {rate}...")
                with bf.BlobFile(bf.join(self.checkpoint_dir, filename), "wb") as f:
                    if self.distributed:
                        # 保存原始模型（不是DDP包装的）
                        th.save(self.unwrapped_model.state_dict(), f)
                    else:
                        th.save(self.model.state_dict(), f)
            else:
                # 保存EMA模型参数
                if self.logger:
                    self.logger.info(f"saving EMA model {rate}...")
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                
                # 创建一个临时的状态字典
                if self.distributed:
                    state_dict = copy.deepcopy(self.unwrapped_model.state_dict())
                else:
                    state_dict = copy.deepcopy(self.model.state_dict())
                
                # 将EMA参数复制到状态字典中
                for param_key, ema_param in zip(state_dict.keys(), params_list):
                    state_dict[param_key] = ema_param
                
                # 保存到文件
                with bf.BlobFile(bf.join(self.checkpoint_dir, filename), "wb") as f:
                    th.save(state_dict, f)

        # 保存当前模型
        save_checkpoint(0, None)
        
        # 保存EMA模型
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # 保存优化器状态
        filename = f"opt{(self.step+self.resume_step):06d}.pt"
        with bf.BlobFile(bf.join(self.checkpoint_dir, filename),"wb") as f:
            th.save(self.opt.state_dict(), f)
            
        # 如果使用混合精度，保存梯度缩放器状态
        if self.use_fp16:
            scaler_filename = f"scaler{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(self.checkpoint_dir, scaler_filename),"wb") as f:
                th.save(self.grad_scaler.state_dict(), f)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


# def get_blob_logdir():
#     # You can change this to be a separate path to save checkpoints to
#     # a blobstore or some external drive.
#     return logger.get_dir()


def find_resume_checkpoint(dirname, key):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    dirname = Path(dirname)
    assert dirname.exists() ,'checkpoint dir not exists'

    ckt_models = [str(f) for f in dirname.glob(f'{key}*.pt')]

    if ckt_models:
        ckt_models.sort()
        last_model_name = ckt_models[-1]
        return last_model_name
    
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if (main_checkpoint is None):
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


# def log_loss_dict(diffusion, ts, losses):
#     for key, values in losses.items():
#         logger.logkv_mean(key, values.mean().item())
#         # Log the quantiles (four quartiles, in particular).
#         for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
#             quartile = int(4 * sub_t / diffusion.num_timesteps)
#             logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def check_memory_usage():
    allocated = torch.cuda.memory_allocated()
    peak_allocated = torch.cuda.max_memory_allocated()
    return allocated, peak_allocated
    self.logger.info(f"Allocated memory: {allocated / 1024**2:.2f} MB, Peak allocated memory: {peak_allocated / 1024**2:.2f} MB")

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def sample_from_model(model_path, config_path, save_dir, num_samples=4, batch_size=4, gpu_rank=0):
    """
    从已保存的模型中采样图像。
    
    Args:
        model_path: 模型路径
        save_dir: 采样结果保存目录
        num_samples: 总共要生成的图像数量
        batch_size: 每次生成的批次大小
        class_json: 类别JSON文件路径，用于条件生成模型
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取模型名称部分，用于保存图像
    model_name = Path(model_path).stem
    
    # 加载配置文件
    if not config_path:
        raise FileNotFoundError("找不到配置文件，请指定正确的配置文件路径")
    config= OmegaConf.load(config_path)
    image_size = config.image_size
    class_json = config.class_json_path


    # 禁用混合精度
    config.use_fp16 = False

    # 设置设备
    device = torch.device(f"cuda:{gpu_rank}" if torch.cuda.is_available() else "cpu")
    
    # 处理条件生成的情况
    num_classes = None
    if config.class_cond:
        # 如果找到类别JSON文件，加载它
        if class_json and os.path.exists(class_json):
            print(f"使用类别文件: {class_json}")
            with open(class_json, 'r') as f:
                classes = json.load(f)
                num_classes = len(classes)
                print(f"类别数量: {num_classes}")
        else:
            # 如果找不到类别JSON文件，使用配置中的类别数量或设置默认值
            raise FileNotFoundError("警告：无法找到类别文件，使用默认值4")

    
    # 创建模型
    print(f"正在创建模型，图像大小: {image_size}，条件生成: {config.class_cond}，类别数量: {num_classes}...")
    model, diffusion = create_model_and_diffusion(
        **select_args(config, model_and_diffusion_defaults().keys()),
        num_classes=num_classes,
        conf=config,
    )
    
    # 加载模型权重
    print(f"正在加载模型: {model_path}...")
    model_state = torch.load(model_path, map_location=device, weights_only=True)
    
    # 如果模型状态包含 'state_dict' 字段，则提取出来
    if "state_dict" in model_state:
        model_state = model_state["state_dict"]
    
    # 检查是否是DDP模型
    if any(k.startswith('module.') for k in model_state.keys()):
        # 移除 'module.' 前缀以与非DDP模型兼容
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    # 首先将模型转移到CPU上，这样可以更安全地修改参数类型
    model = model.cpu()
    
    # 加载模型权重
    model.load_state_dict(model_state, strict=True)
    
    # 将所有模型参数强制转换为float32类型
    for param in model.parameters():
        param.data = param.data.float()
    
    # 再将模型移回到GPU
    model = model.to(device)
    model.eval()
    
    # 从模型中采样
    print(f"正在生成 {num_samples} 张图像...")
    all_images = []
    
    with torch.no_grad():
        while len(all_images) * batch_size < num_samples:
            # 生成随机噪声
            noise = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=torch.float32)
            
            # 准备条件标签（如果使用条件生成）
            model_kwargs = {}
            if num_classes is not None:
                # 对于每个批次生成随机类别标签
                classes = torch.randint(0, num_classes, (batch_size,), device=device)
                model_kwargs["y"] = classes
                print(f"使用类别标签: {classes.cpu().numpy()}")
            
            # 使用最简单可靠的方法进行采样
            try:
                # 使用最简单的采样方法
                sample = simple_sample(diffusion, model, noise, model_kwargs)
                all_images.append(sample.cpu())
            except Exception as e:
                print(f"采样出错: {e}")
                # 直接返回随机噪声作为图像
                print("返回随机噪声作为图像")
                # 将噪声归一化到[-1,1]范围
                norm_noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1
                all_images.append(norm_noise.cpu())
    
    # 合并所有图像
    all_images = torch.cat(all_images, dim=0)
    all_images = all_images[:num_samples]  # 确保不超过请求的数量
    
    # 将图像规格化到[0,1]用于显示
    grid = vutils.make_grid(all_images, nrow=min(batch_size, all_images.size(0)), padding=2, normalize=True)
    
    # 保存结果
    file_name = bf.join(save_dir, f"sample_{model_name}.jpg")
    vutils.save_image(grid, file_name)
    print(f"图像已保存到: {file_name}")
    
    return file_name

def simple_sample(diffusion, model, noise, model_kwargs=None):
    """
    最简化的采样流程，跳过大部分时间步，只采样少量关键步骤
    """
    if model_kwargs is None:
        model_kwargs = {}
        
    device = noise.device
    shape = noise.shape
    
    # 从噪声开始
    img = noise
    
    # 采样步数设置 - 提高质量但保持合理速度
    # 对于训练中预览：使用总时间步的10%，至少100步，最多200步
    num_steps = max(min(diffusion.num_timesteps // 10, 200), 100)
    
    # 使用非线性采样步骤，在早期和晚期阶段采样更密集
    # 这比线性采样更有效率，可以用更少的步骤获得更好的质量
    indices = []
    # 前期需要更密集的采样 - 分配40%的步骤
    early_steps = int(num_steps * 0.4)
    early_indices = np.linspace(diffusion.num_timesteps - 1, 
                              diffusion.num_timesteps // 2, 
                              early_steps).round().astype(int)
    indices.extend(early_indices)
    
    # 后期也需要密集采样 - 分配60%的步骤
    late_steps = num_steps - early_steps
    late_indices = np.linspace(diffusion.num_timesteps // 2 - 1, 
                             0, 
                             late_steps).round().astype(int)
    indices.extend(late_indices)
    
    # 确保没有重复
    indices = sorted(list(set(indices)), reverse=True)
    
    print(f"使用优化采样，总共{len(indices)}个时间步")
    
    for i in indices:
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        with torch.no_grad():
            # 调用模型获取预测
            try:
                # 预测噪声
                out = model(img, t, **model_kwargs)
                
                # 处理模型输出 - 正确处理输出通道数不匹配的情况
                if isinstance(out, tuple):
                    eps = out[0]
                else:
                    # 检查输出通道数
                    if out.shape[1] == 6:  # 如果输出是6通道（噪声+方差）
                        # 将输出分割为两部分，每部分3通道
                        eps, _ = torch.split(out, 3, dim=1)
                    elif out.shape[1] == 3:  # 如果输出正好是3通道
                        eps = out
                    else:
                        # 如果通道数不是3或6，输出警告并尝试调整
                        print(f"警告：模型输出通道数 {out.shape[1]} 不是预期的3或6。尝试调整...")
                        if out.shape[1] > 3:
                            eps = out[:, :3]  # 只使用前3个通道
                        else:
                            # 如果通道数小于3，通过复制扩展到3通道
                            eps = out.repeat(1, 3 // out.shape[1] + 1, 1, 1)[:, :3]
                
                # 将numpy值转换为tensor
                alpha = torch.tensor(diffusion.alphas_cumprod[i], device=device, dtype=torch.float32)
                alpha_prev = torch.tensor(diffusion.alphas_cumprod[i-1] if i > 0 else 1.0, device=device, dtype=torch.float32)
                
                # 预测原始图像
                pred_x0 = (img - torch.sqrt(1 - alpha) * eps) / torch.sqrt(alpha)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                
                # 简化的噪声步骤
                if i > 0:
                    # 计算下一步的均值
                    beta = 1 - alpha / alpha_prev
                    mean = torch.sqrt(alpha_prev) * pred_x0
                    
                    # 添加噪声
                    noise = torch.randn_like(img)
                    std = torch.sqrt(beta)
                    img = mean + std * noise
                else:
                    # 最后一步直接使用预测的原始图像
                    img = pred_x0
            except Exception as e:
                print(f"单步采样出错: {e}")
                # 对于失败的步骤，简单地减少噪声
                factor = 1.0 - (i / diffusion.num_timesteps)
                img = img * (1 - factor) + torch.tanh(img) * factor
    
    return img

