from functools import partial
import math
import os
import random
from tkinter import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm
from torchvision.transforms import functional as TF

from utils.logger import logging_info
from utils.drawer import visualize_and_save_attention_process
from .gaussian_diffusion import _extract_into_tensor
from .new_scheduler import ddim_timesteps, ddim_repaint_timesteps
from .respace import SpacedDiffusion
from .losses import  original_loss
from skimage import measure


def compute_attention_masks(pred_crop0, img_free_crop=None, 
                           temperature=0.1, blur_sigma=1.0, attn_defect_accum=None,
                           center_weight=0.6, radial_decay=0.4, connected_components=True, 
                           min_area_ratio=0.01, center_region_ratio=0.25, adaptive_threshold=True, 
                           threshold_factor=0.4, accumulation_beta=0.95):
    """
    Compute attention masks based on pixel differences with optional background filtering.
    
    Args:
        pred_crop0: Defect model prediction
        img_free_crop: Defect-free background crop
        temperature: Attention sharpness coefficient
        blur_sigma: Gaussian blur sigma
        attn_defect_accum: Accumulated attention (EMA)
        center_weight: Center weighting (0-1)
        radial_decay: Radial decay rate (0-1)
        connected_components: Enable component filtering
        min_area_ratio: Min component area ratio
        center_region_ratio: Center region size (0-1)
        adaptive_threshold: Use adaptive thresholding
        threshold_factor: Threshold scale (0-1)
        accumulation_beta: EMA beta
        
    Returns:
        attn_defect, attn_bg, attn_defect_instant
    """
    
    # Compute and normalize pixel difference
    pixel_diff = torch.abs(img_free_crop - pred_crop0)
    if pixel_diff.shape[1] > 1: 
        pixel_diff = pixel_diff.mean(dim=1, keepdim=True)
        
    batch_size, _, height, width = pixel_diff.shape
    pixel_diff_flat = pixel_diff.view(batch_size, -1)
    min_vals = pixel_diff_flat.min(dim=1, keepdim=True)[0]
    max_vals = pixel_diff_flat.max(dim=1, keepdim=True)[0]
    range_vals = max_vals - min_vals + 1e-8
    pixel_diff_normalized = (pixel_diff_flat - min_vals) / range_vals
    pixel_diff_normalized = pixel_diff_normalized.view(batch_size, 1, height, width)

    # Enhance contrast with sigmoid
    attn_defect_raw = torch.sigmoid((pixel_diff_normalized - 0.5) / temperature)
    
    # Apply center weighting
    if center_weight > 0:
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, height, device=attn_defect_raw.device),
            torch.linspace(-1, 1, width, device=attn_defect_raw.device)
        )
        dist_from_center = torch.sqrt(x_grid**2 + y_grid**2)
        dist_from_center = dist_from_center / torch.max(dist_from_center)
        center_mask = torch.exp(-(dist_from_center**2) / (2 * radial_decay**2))
        center_mask = center_mask.unsqueeze(0).unsqueeze(0).expand_as(attn_defect_raw)
        
        weighted_attention = attn_defect_raw * (1.0 + center_weight * center_mask)
        attn_defect_raw = torch.clamp(weighted_attention, 0, 1)
    
    # Apply Gaussian blur
    if blur_sigma > 0:
        blur_kernel_size = max(3, int(blur_sigma * 3) * 2 + 1)
        attn_defect_blurred = torch.zeros_like(attn_defect_raw)
        for b in range(attn_defect_raw.shape[0]):
            attn_defect_blurred[b] = TF.gaussian_blur(
                attn_defect_raw[b],
                kernel_size=[blur_kernel_size, blur_kernel_size],
                sigma=[blur_sigma, blur_sigma]
            )
        attn_defect_instant = attn_defect_blurred
    else:
        attn_defect_instant = attn_defect_raw
        
    # Connected component filtering
    if connected_components:
        filtered_mask = torch.zeros_like(attn_defect_instant)
            
        # Create Gaussian center mask
        y_grid_np, x_grid_np = np.meshgrid(
            np.linspace(-1, 1, height),
            np.linspace(-1, 1, width),
            indexing='ij'
        )
        center_radius = center_region_ratio * np.sqrt(2)
        center_region_mask_soft = np.exp(-(x_grid_np**2 + y_grid_np**2) / (2 * (center_radius/2.5)**2))
        center_region_mask = center_region_mask_soft > 0.5
        
        for b in range(batch_size):
            mask_np = attn_defect_instant[b, 0].cpu().numpy()
            
            # Compute threshold
            if adaptive_threshold:
                center_values = mask_np[center_region_mask]
                center_mean = np.mean(center_values)
                threshold = max(0.05, min(center_mean * threshold_factor, 0.8))
                logging_info(f"Adaptive threshold: {threshold:.3f} (center mean: {center_mean:.3f})")
            else:
                threshold = 0.2
            
            # Prevent all-black mask
            binary_test = (mask_np > threshold).astype(np.uint8)
            if np.sum(binary_test) == 0:
                sorted_values = np.sort(mask_np.flatten())
                idx = max(0, int(len(sorted_values) * 0.95) - 1)
                threshold = min(sorted_values[idx], 0.2)
                logging_info(f"Preventing all-black mask: adjusted threshold to {threshold:.3f}")
            
            binary_mask = (mask_np > threshold).astype(np.uint8)
            
            # Label and filter components
            labeled_mask = measure.label(binary_mask, connectivity=2)
            props = measure.regionprops(labeled_mask)
            min_area = height * width * min_area_ratio
            
            valid_mask = np.zeros_like(binary_mask, dtype=np.float32)
            for prop in props:
                region_mask = labeled_mask == prop.label
                intersects_center = np.any(region_mask & center_region_mask)
                
                if prop.area >= min_area and intersects_center:
                    valid_mask[region_mask] = mask_np[region_mask]
                    
            if np.sum(valid_mask) == 0:
                logging_info(f"No valid regions found, using smooth decay mask")
                valid_mask = mask_np * center_region_mask_soft
            
            filtered_mask[b, 0] = torch.from_numpy(valid_mask).to(attn_defect_instant.device)
        
        attn_defect_instant = filtered_mask
    
    attn_defect_instant = torch.clamp(attn_defect_instant, 0, 1)
    
    # Apply EMA if accumulated mask provided
    if attn_defect_accum is not None:
        attn_defect_final = attn_defect_accum * accumulation_beta + attn_defect_instant * (1 - accumulation_beta)
        attn_defect_final = (attn_defect_final - attn_defect_final.min()) / (attn_defect_final.max() - attn_defect_final.min() + 1e-8)
    else:
        attn_defect_final = attn_defect_instant
        
    attn_bg_final = (attn_defect_final < 0.01).float()

    return attn_defect_final, attn_bg_final, attn_defect_instant



def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def get_g_scha(length = 250, start = 1, end = 10,  stop=0):
    """
    Generate a gradient schedule sequence.
    
    Args:
        length: Total length of sequence
        start: Starting value
        end: Ending value
        stop: Number of zeros to prepend
    """
    # More precise step calculation considering floating point
    step = ((end - start) / (length - stop - 1)) 

    # Generate sequence through accumulation and rounding
    sequence = [round(start + step * i) for i in range(length - stop)]

    # Ensure last value is the target end value
    if sequence[-1] != end:
        sequence[-1] = end
    
    if stop:
        sequence = [0] * stop + sequence

    return sequence


class DDIMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.ddim_sigma = conf.get("ddim.ddim_sigma")

    def _get_et(self, model_fn, x, t, model_kwargs):
        model_fn = self._wrap_model(model_fn)
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, _ = torch.split(model_output, C, dim=1)
        return model_output

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        with torch.no_grad():
            alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, prev_t, x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )

            def process_xstart(_x):
                if denoised_fn is not None:
                    _x = denoised_fn(_x)
                if clip_denoised:
                    return _x.clamp(-1, 1)
                return _x

            e_t = self._get_et(model_fn, x, t, model_kwargs)
            pred_x0 = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=e_t))

            mean_pred = (
                pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * e_t
            )
            noise = noise_like(x.shape, x.device, repeat=False)

            nonzero_mask = (t != 0).float().view(-1, *
                                                ([1] * (len(x.shape) - 1)))
            x_prev = mean_pred + noise * sigmas * nonzero_mask

        return {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
        }

    def q_sample_middle(self, x, cur_t, tar_t, no_noise=False):
        assert cur_t <= tar_t
        device = x.device
        while cur_t < tar_t:
            if no_noise:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            _cur_t = torch.tensor(cur_t, device=device)
            beta = _extract_into_tensor(self.betas, _cur_t, x.shape)
            x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
            cur_t += 1
        return x

    def q_sample(self, x_start, t, no_noise=False, noise=None):
        if no_noise:
            noise = torch.zeros_like(x_start)
        else:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod,
                                t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def x_forward_sample(self, x0, forward_method="from_0", no_noise=False):
        x_forward = [self.q_sample(x0, torch.tensor(0, device=x0.device))]
        if forward_method == "from_middle":
            for _step in range(0, len(self.timestep_map) - 1):
                x_forward.append(
                    self.q_sample_middle(
                        x=x_forward[-1][0].unsqueeze(0),
                        cur_t=_step,
                        tar_t=_step + 1,
                        no_noise=no_noise,
                    )
                )
        elif forward_method == "from_0":
            for _step in range(1, len(self.timestep_map)):
                x_forward.append(
                    self.q_sample(
                        x_start=x0[0].unsqueeze(0),
                        t=torch.tensor(_step, device=x0.device),
                        no_noise=no_noise,
                    )
                )
        return x_forward

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])
        time_pairs = list(zip(steps[:-1], steps[1:]))

        x0 = model_kwargs["gt"]
        x_forwards = self.x_forward_sample(x0)
        mask = model_kwargs["gt_keep_mask"]

        x_t = img
        import os
        from utils import normalize_image, save_grid

        for cur_t, prev_t in tqdm(time_pairs):
            # Replace surrounding region
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            cur_t = torch.tensor([cur_t] * shape[0], device=device)
            prev_t = torch.tensor([prev_t] * shape[0], device=device)

            output = self.p_sample(
                model_fn,
                x=x_t,
                t=cur_t,
                prev_t=prev_t,
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )
            x_t = output["x_prev"]

            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles",
                                f"mid-{prev_t[0].item()}.png"),
                )
                save_grid(
                    normalize_image(output["pred_x0"]),
                    os.path.join(sample_dir, "middles",
                                f"pred-{prev_t[0].item()}.png"),
                )

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }

class A_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        assert conf.get("optimize_xt.optimize_xt",
                        False), "Double check on optimize"
        self.ddpm_num_steps = conf.get(
            "ddim.schedule_params.ddpm_num_steps", 250)
        self.coef_xt_reg = conf.get("optimize_xt.coef_xt_reg", 0.001)
        self.coef_xt_reg_decay = conf.get("optimize_xt.coef_xt_reg_decay", 1.0)
        self.num_iteration_optimize_xt = conf.get(
            "optimize_xt.num_iteration_optimize_xt", 1
        )
        self.lr_xt = conf.get("optimize_xt.lr_xt", 0.001)
        self.lr_xt_decay = conf.get("optimize_xt.lr_xt_decay", 1.0)
        self.use_smart_lr_xt_decay = conf.get(
            "optimize_xt.use_smart_lr_xt_decay", False
        )
        self.use_adaptive_lr_xt = conf.get(
            "optimize_xt.use_adaptive_lr_xt", False)
        self.mid_interval_num = int(conf.get("optimize_xt.mid_interval_num", 1))
        if conf.get("ddim.schedule_params.use_timetravel", False):
            self.steps = ddim_repaint_timesteps(**conf["ddim.schedule_params"])
        else:
            self.steps = ddim_timesteps(**conf["ddim.schedule_params"])
        
        self.mode = conf.get("mode", "inpaint")
        self.scale = conf.get("scale", 0)
        self.sqrt_alpcu = torch.from_numpy(self.sqrt_alphas_cumprod)

        # Cache configuration parameters
        self.crop_size = conf['crop']['image_size']
        self.lr_crop = conf["optimize_xt"]["lr_crop"]


    def p_sample(
        self,
        model_fn,
        guide_models,
        x,
        t,
        prev_t,
        model_kwargs,
        lr_xt,
        coef_xt_reg,
        conf,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        def loss_fn(pred_x0_crop, pred_crop0, pred_x0, img_free, img_free_crop=None, bg_mask=None, attn_defect=None, attn_bg=None, logg=None):
            if attn_defect is not None and attn_bg is not None:
                # Defect region: encourage similarity with defect prediction
                defect_loss_term = torch.mean(attn_defect * (pred_x0_crop - pred_crop0)**2) * 1e5
                
                # Background region: encourage similarity with defect-free image
                if img_free_crop is None:
                    bg_loss_term = torch.mean(attn_bg * (pred_x0_crop - pred_x0_crop.detach())**2)
                else:
                    bg_loss_term = torch.mean(attn_bg * (pred_x0_crop - img_free_crop)**2) * 1e4

                weighted_crop_loss = defect_loss_term + bg_loss_term
                
                # Outside region: maintain consistency with defect-free image
                outside_loss = torch.mean(bg_mask * (pred_x0 - img_free)**2) * 5e4
                
                total_loss = weighted_crop_loss + outside_loss
            else:
                # Original loss function logic
                defect_loss = original_loss(pred_x0_crop, pred_crop0) * 10.0
                bg_loss = original_loss(pred_x0, img_free, bg_mask) * 0.06
                total_loss = defect_loss + bg_loss
            
            # Logging total loss
            if logg:
                log_msg = f"total_loss: {total_loss.item():.4f}"
                if attn_defect is not None:
                    log_msg = f"crop_defect_loss: {defect_loss_term.item():.4f}\t" + \
                              f"crop_bg_loss: {bg_loss_term.item():.4f}\t" + \
                              f"outside_loss: {outside_loss.item():.4f}\t" + log_msg
                else:
                    log_msg = f"defect_loss: {defect_loss.item():.4f}\t" + \
                              f"bg_loss: {bg_loss.item():.4f}\t" + log_msg
                logging_info(log_msg)

            return total_loss

        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1, 1)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                res = grad_ckpt(
                    self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False
                )
            else:
                res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res
    
        def get_smart_lr_decay_rate(_t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num

            # Get reversed list with interval_num spacing, head is current t, tail is 0
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)

            ret = 1
            time_pairs = list(zip(steps[:-1], steps[1:]))
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(
                    self.alphas_cumprod[_prev_t]
                )
            return ret

        def multistep_predx0(_x, _et, _t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            time_pairs = list(zip(steps[:-1], steps[1:]))
            x_t = _x
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                _cur_t = torch.tensor([_cur_t] * _x.shape[0], device=_x.device)
                _prev_t = torch.tensor(
                    [_prev_t] * _x.shape[0], device=_x.device)
                if i != 0:
                    _et = get_et(x_t, _cur_t)
                x_t = grad_ckpt(
                    get_update, x_t, _cur_t, _prev_t, _et, None, use_reentrant=False
                )
            return x_t

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))
            else:
                _pred_x0 = grad_ckpt(
                    multistep_predx0, _x, _et, _t, interval_num, use_reentrant=False
                )
                return process_xstart(_pred_x0)
        
        def get_update(  # x_t -> x_t-1
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                    *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        B, C = x.shape[:2]
        assert t.shape == (B,)
        crop = model_kwargs["crop_t"]
        mask = model_kwargs["gt_keep_mask"]
        location = model_kwargs["location"]
        bg_mask = 1.0 - mask
        img_free = model_kwargs["img_free"]
        crop_model = guide_models

        # Configure attention feature extraction parameters
        temperature = conf["attention_features"]["temperature"]
        blur_sigma = conf["attention_features"]["blur_sigma"]
        use_attention = conf["attention_features"]["enabled"]
        
        # Attention accumulation configuration
        accumulation_enabled = conf["attention_features"]["accumulation_enabled"]
        accumulation_beta = conf["attention_features"]["accumulation_beta"]
        adaptive_beta = conf["attention_features"]["adaptive_beta"]
        beta_min = conf["attention_features"]["beta_min"]
        
        # If adaptive beta enabled, adjust beta based on timestep
        if adaptive_beta:
            # t decreases from large to small, beta should decrease (early: more memory, later: more current)
            max_t = conf["ddim"]["schedule_params"]["ddpm_num_steps"]
            # Linear interpolation to calculate current beta
            t_val = t[0].item()
            beta_ratio = t_val / max_t
            accumulation_beta = beta_min + (accumulation_beta - beta_min) * beta_ratio
            logging_info(f"Adaptive beta at t={t_val}: {accumulation_beta:.4f}")

        # Condition mean
        if cond_fn is not None:
            model_fn = self._wrap_model(model_fn)
            B, C = x.shape[:2]
            assert t.shape == (B,)
            model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            _, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
            with torch.enable_grad():
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                x = x + model_variance * gradient

        # Learning rate
        if self.use_smart_lr_xt_decay:
            lr_xt /= get_smart_lr_decay_rate(t, self.mid_interval_num)
            lr_cropt = lr_xt if self.lr_crop else 0.0

        # Optimize
        with torch.enable_grad():
            # Clone and detach inputs once to preserve original states
            origin_x = x.detach().clone()
            origin_crop = crop.detach().clone()

            # Enable gradients for optimization
            x = origin_x.detach().requires_grad_(True)
            crop = origin_crop.detach().requires_grad_(True)

            # Precompute crop indices to avoid recalculating in loops
            crop_y_start, crop_x_start = location[1], location[0]
            crop_size = conf['crop']['image_size']
            crop_y_end = crop_y_start + crop_size
            crop_x_end = crop_x_start + crop_size

            # Perform initial predictions for the large image
            e_t = get_et(_x=x, _t=t)
            pred_x0 = get_predx0(
                _x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num
            )
            pred_x0_crop = pred_x0[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            # Perform initial predictions for the small image (crop)
            crop_e_t = self._get_et(
                crop_model, crop, t, model_kwargs={"y": model_kwargs["crop_y"]}
            )
            pred_crop0 = get_predx0(_x=crop, _t=t, _et=crop_e_t)
            
            # Compute attention masks - calculate once before optimization, shared for entire timestep
            img_free_crop, attn_defect, attn_bg, attn_defect_instant = None, None, None, None
            # Extract real background crop region
            img_free_crop = img_free[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end].detach()
            
            if use_attention:
                # Check if attention accumulation is enabled
                attn_defect_accum = model_kwargs.get("attn_defect_accum", None)

                # Get background recognition threshold and filter switch
                start_step = conf["ddim"]["schedule_params"]["start_step"]
                
                # Compute attention masks based on feature similarity
                attn_defect, attn_bg, attn_defect_instant = compute_attention_masks(
                    pred_crop0=pred_crop0.detach(),
                    img_free_crop=img_free_crop,
                    temperature=temperature,
                    blur_sigma=blur_sigma,
                    attn_defect_accum=attn_defect_accum if accumulation_enabled else None,
                    center_weight=conf["attention_features"]["center_weight"],
                    radial_decay=conf["attention_features"]["radial_decay"],
                    connected_components=conf["attention_features"]["connected_components"],
                    min_area_ratio=conf["attention_features"]["min_area_ratio"],
                    center_region_ratio=conf["attention_features"]["center_region_ratio"],
                    adaptive_threshold=conf["attention_features"]["adaptive_threshold"],
                    threshold_factor=conf["attention_features"]["threshold_factor"],
                    accumulation_beta=accumulation_beta,
                )

                logging_info(f"Accumulated raw attention masks computed: defect max={attn_defect.max().item():.3f}, min={attn_defect.min().item():.3f}")

            # Compute initial loss between large and small image predictions
            prev_loss = loss_fn(
                pred_x0_crop, pred_crop0, pred_x0, img_free, img_free_crop, bg_mask, 
                attn_defect, attn_bg
            ).item()

            # Logging current step and learning rates
            step_num = t[0].item()
            logging_info(f"step: {step_num} \t"  \
                        f"lr_xt {lr_xt:.8f} lr_cropt {lr_cropt:.8f}")

            # Retrieve the number of gradient steps for the current timestep
            step_count = get_g_scha(length=self.num_timesteps, **conf.g_scha)[t[0]]

            # Precompute regularization coefficient
            coef_reg = coef_xt_reg

            # Precompute static parts of the regularization term
            reg_fn_origin_x = lambda x_new: reg_fn(origin_x, x_new)
            reg_fn_origin_crop = lambda crop_new: reg_fn(origin_crop, crop_new)

            # Iterate over the number of gradient steps
            for _ in range(step_count):
                # Compute loss with regularization
                reg_term = reg_fn_origin_x(x) + reg_fn_origin_crop(crop)
                loss = loss_fn(
                    pred_x0_crop, pred_crop0, pred_x0, img_free, img_free_crop,  bg_mask, 
                    attn_defect, attn_bg, logg=True
                ) + coef_reg * reg_term

                # Compute gradients with respect to x and crop
                x_grad, crop_grad = torch.autograd.grad(
                    outputs=loss,
                    inputs=[x, crop],
                    retain_graph=False,
                    create_graph=False,
                )

                # Update x and crop using gradient descent
                new_x = x - lr_xt * x_grad.detach()
                new_crop = crop - lr_cropt * crop_grad.detach()

                # Adaptive learning rate adjustment if enabled
                if self.use_adaptive_lr_xt:
                    while True:
                        with torch.no_grad():
                            # Compute predictions with updated x and crop
                            e_t_new = get_et(_x=new_x, _t=t)
                            pred_x0_new = get_predx0(
                                _x=new_x, _t=t, _et=e_t_new, interval_num=self.mid_interval_num
                            )
                            pred_x0_crop_new = pred_x0_new[
                                :, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end
                            ]

                            crop_e_t_new = self._get_et(
                                crop_model, new_crop, t, model_kwargs={"y": model_kwargs["crop_y"]}
                            )
                            pred_crop0_new = get_predx0(_x=new_crop, _t=t, _et=crop_e_t_new)

                            # Compute new loss with updated predictions
                            reg_new = reg_fn_origin_x(new_x) + reg_fn_origin_crop(new_crop)
                            new_loss = (
                                loss_fn(pred_x0_crop_new, pred_crop0_new, pred_x0_new, img_free, img_free_crop,  bg_mask, attn_defect, attn_bg).item()
                                + coef_reg * reg_new
                            )

                            # Check if the new loss is acceptable
                            if not math.isnan(new_loss) and new_loss <= loss.item():
                                break  # Accept the update
                            else:
                                # Reduce learning rates by a factor of 0.8
                                lr_xt *= 0.8
                                lr_cropt *= 0.8
                                logging_info(
                                    f"Loss too large ({loss.item():.3f}->{new_loss:.3f})! "
                                    f"Learning rate decreased to {lr_xt:.5f}."
                                )
                                # Recompute updates with the reduced learning rates
                                new_x = x - lr_xt * x_grad.detach()
                                new_crop = crop - lr_cropt * crop_grad.detach()

                # Update x and crop for the next iteration, enabling gradients
                x = new_x.detach().requires_grad_(True)
                crop = new_crop.detach().requires_grad_(True)

                # Recompute predictions with the updated x and crop
                e_t = get_et(_x=x, _t=t)
                pred_x0 = get_predx0(
                    _x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                )
                pred_x0_crop = pred_x0[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

                crop_e_t = self._get_et(
                    crop_model, crop, t, model_kwargs={"y": model_kwargs["crop_y"]}
                )
                pred_crop0 = get_predx0(_x=crop, _t=t, _et=crop_e_t)

                # Clean up to free memory
                del loss, x_grad, crop_grad
                torch.cuda.empty_cache()

            # After optimization
            with torch.no_grad():
                new_loss = loss_fn(pred_x0_crop, pred_crop0, pred_x0, img_free, img_free_crop, bg_mask, attn_defect, attn_bg).item()
                logging_info(f"After update, loss change: {prev_loss:.3f} -> {new_loss:.3f}")

                if 'new_x' in locals() and new_x is not None:
                    new_reg_x = reg_fn(origin_x, new_x).item()
                    logging_info(f"X regularization change: 0.000 -> {new_reg_x:.3f}")
                if 'new_crop' in locals() and new_crop is not None:
                    new_reg_crop = reg_fn(origin_crop, new_crop).item()
                    logging_info(f"Crop regularization change: 0.000 -> {new_reg_crop:.3f}")

                # Detach tensors to prevent gradient tracking
                pred_x0 = pred_x0.detach()
                e_t = e_t.detach()
                x = x.detach()
                pred_crop0 = pred_crop0.detach()
                crop_e_t = crop_e_t.detach()
                crop = crop.detach()

                # Clean up to free memory
                del origin_x, origin_crop, prev_loss

        if t[0] / 10 == 0:
            with torch.no_grad():
                crop = self.q_sample(
                    x_start=pred_x0_crop,
                    t=t,
                )
                crop_e_t = self._get_et(
                    crop_model, crop, t, model_kwargs={"y": model_kwargs["crop_y"]}
                )
                pred_crop0 = get_predx0(_x=crop, _t=t, _et=crop_e_t)
                pred_crop0 = pred_crop0.detach()
                crop_e_t = crop_e_t.detach()
                crop = crop.detach()

        # Update previous states
        x_prev = get_update(
            x,
            t,
            prev_t,
            e_t,
            _pred_x0=pred_x0 if self.mid_interval_num == 1 else None,
        )
        crop_prev = get_update(
            crop,
            t,
            prev_t,
            crop_e_t,
            _pred_x0=pred_crop0 if self.mid_interval_num == 1 else None,
        )
        del e_t, crop_e_t

        return {
            "x": x, 
            "x_prev": x_prev, 
            "pred_x0": pred_x0, 
            "crop": crop,
            "crop_prev": crop_prev,
            "pred_crop0": pred_crop0,
            "loss": new_loss, 
            "attn_defect": attn_defect,
            "attn_bg": attn_bg,
            "attn_defect_instant": attn_defect_instant,
        }
    
    def p_sample_loop(
            self,
            model_fn,
            shape,
            guide_models,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            return_all=False,
            conf=None,
            start_step=None,
            **kwargs,
        ):
            """
            Generate samples from given noise, supporting denoising from specific timestep.
            
            Args:
                model_fn: Model function
                shape: Shape of generated image
                guide_models: Guide models (crop region model)
                noise: Optional starting noise, generates random noise if None
                clip_denoised: Whether to clip denoised results to [-1,1]
                denoised_fn: Optional post-denoising function
                cond_fn: Condition function
                model_kwargs: Additional model parameters
                device: Device
                progress: Whether to show progress bar
                return_all: Whether to return all intermediate results
                conf: Configuration
                start_step: Starting timestep for denoising
            """
            if device is None:
                device = next(model_fn.parameters()).device
            assert isinstance(shape, (tuple, list)), "Shape must be a tuple or list"
            
            # Unpack location coordinates
            loc_x, loc_y = model_kwargs["location"]  # location is [x, y]
            
            # Use class properties
            crop_size = self.crop_size
            
            # Use provided noise image
            x_t = noise
            crop_t = model_kwargs["crop_t"]
            
            # Set hyperparameters
            lr_xt = self.lr_xt
            coef_xt_reg = self.coef_xt_reg
            
            # Initialize variables
            loss = None
            
            # Create gradient descent mask function
            cx = loc_x + crop_size // 2
            cy = loc_y + crop_size // 2
            
            # Precompute crop region end coordinates
            crop_end_x = loc_x + crop_size
            crop_end_y = loc_y + crop_size

            # Initialize gt_keep_mask
            gt_keep_mask = torch.zeros_like(x_t)
            gt_keep_mask[:, :, loc_y:crop_end_y, loc_x:crop_end_x] = 1.0
            model_kwargs["gt_keep_mask"] = gt_keep_mask
            model_kwargs["crop_t"] = crop_t
            
            status = None
            x_t_process = []
            crop_process = []
            progress_t = 0
            
            # Initialize dictionary to store attention masks
            attention_masks_dict = {}
            use_attention = conf["attention_features"]["enabled"]
            model_kwargs["attn_defect_accum"] = None

            # Modify time_pairs to support starting from specific step
            time_pairs = list(zip(self.steps[:-1], self.steps[1:]))
            loop = tqdm(time_pairs)
            
            # Add timestamp for image naming
            import time
            timestamp = int(time.time())
            model_kwargs["image_name"] = model_kwargs.get("image_base_name", f"sample_{timestamp}")
            
            for cur_t, prev_t in loop:
                
                if cur_t > prev_t:  # Denoising phase
                    status = "reverse"
                    
                    model_kwargs["crop_t"] = crop_t
                    
                    # Create timestep tensors
                    t_tensor = torch.full((shape[0],), cur_t, device=device, dtype=torch.long)
                    prev_t_tensor = torch.full((shape[0],), prev_t, device=device, dtype=torch.long)
                    
                    # Execute sampling step
                    output = self.p_sample(
                        model_fn=model_fn,
                        guide_models=guide_models,
                        x=x_t,
                        t=t_tensor,
                        prev_t=prev_t_tensor,
                        model_kwargs=model_kwargs,
                        pred_xstart=None,
                        lr_xt=lr_xt,
                        coef_xt_reg=coef_xt_reg,
                        conf=conf,
                    )
                    x_t = output.get("x_prev")
                    loss = output.get("loss")
                    crop_t = output.get("crop_prev")
                    attn_defect = output.get("attn_defect")
                    attn_bg = output.get("attn_bg")
                    attn_defect_instant = output.get("attn_defect_instant")
                    model_kwargs["attn_defect_accum"] = attn_defect

                    # Save predictions from current output for attention visualization
                    if use_attention and (
                        prev_t == self.steps[1] or  # Add first step of denoising
                        (prev_t < 150 and prev_t % 30 == 0) or 
                        (prev_t < 30 and prev_t % 10 == 0)
                       ):
                        # Compute attention masks for current timestep
                        pred_x0 = output.get("pred_x0")
                        pred_crop0 = output.get("pred_crop0")
                        pred_x0_crop = pred_x0[:, :, loc_y:crop_end_y, loc_x:crop_end_x]
                        
                        # Store in dictionary
                        attention_masks_dict[prev_t] = {
                            'attn_defect': attn_defect.detach(),
                            'attn_bg': attn_bg.detach(),
                            'pred_x0_crop': pred_x0_crop.detach(),
                            'pred_crop0': pred_crop0.detach(),
                            'pred_full': pred_x0.detach(),
                            'attn_defect_instant': attn_defect_instant.detach()
                        }
                        logging_info(f"Saved attention mask for step {prev_t}")
                    
                    # Update process lists
                    if prev_t % 10 == 0:
                        if prev_t == progress_t:
                            x_t_process[-1] = output.get("pred_x0")
                            crop_process[-1] = output.get("pred_crop0")
                        else:
                            x_t_process.append(output.get("pred_x0"))
                            crop_process.append(output.get("pred_crop0"))
                            progress_t = prev_t
                    
                    # Learning rate decay
                    if self.lr_xt_decay != 1.0:
                        logging_info(f"Learning rate decay: {lr_xt:.5f} -> {lr_xt * self.lr_xt_decay:.5f}.")
                    lr_xt *= self.lr_xt_decay
                    if self.coef_xt_reg_decay != 1.0:
                        logging_info(f"Regularization coefficient decay: {coef_xt_reg:.5f} -> {coef_xt_reg * self.coef_xt_reg_decay:.5f}.")
                    coef_xt_reg *= self.coef_xt_reg_decay
                
                else:  # Time travel backward phase
                    if status == "reverse" and conf.get("optimize_xt.optimize_before_time_travel"):
                        # If previous status was reverse, update x_t and crop_t
                        model_kwargs["crop_t"] = crop_t
                        x_t, crop_t = self.get_updated_xt(
                            model_fn=model_fn,
                            guide_models=guide_models,
                            x=x_t,
                            t=torch.full((shape[0],), cur_t, device=device, dtype=torch.long),
                            model_kwargs=model_kwargs,
                            lr_xt=lr_xt,
                            coef_xt_reg=coef_xt_reg,
                            conf=conf,                        
                        )
                    status = "forward"
                    assert prev_t == cur_t + 1, "Only support 1-step time travel back"
                    
                    # Create previous timestep tensor
                    prev_t_tensor = torch.full((shape[0],), prev_t, device=device, dtype=torch.long)
                    with torch.no_grad():
                        x_t = self._undo(x_t, prev_t_tensor)
                        crop_t = self._undo(crop_t, prev_t_tensor)
                    
                    # Restore learning rate
                    logging_info(f"Undo step: {cur_t}")
                    lr_xt /= self.lr_xt_decay
                    coef_xt_reg /= self.coef_xt_reg_decay
                    
            # After completing all timesteps, save attention mask visualizations
            if use_attention and attention_masks_dict:
                # Check if configuration has GIF generation parameter
                generate_gif = conf.get('generate_gif', False)
                visualize_and_save_attention_process(attention_masks_dict, conf, model_kwargs, generate_gif=generate_gif)
                logging_info(f"Saved attention mask visualizations for {len(attention_masks_dict)} timesteps")
            
            # Construct target tensor gt
            gt = torch.zeros_like(x_t)
            gt[:, :, loc_y:crop_end_y, loc_x:crop_end_x] = crop_t
            
            return {
                "sample": x_t, 
                "gt": gt, 
                "loss": loss, 
                "crop_process": crop_process, 
                "x_t_process": x_t_process, 
                "crop": crop_t,
            }

    
    def get_updated_xt(self, model_fn, guide_models, x, t, model_kwargs, lr_xt, coef_xt_reg, conf):
        result = self.p_sample(
            model_fn,
            guide_models,
            x=x,
            t=t,
            prev_t=torch.zeros_like(t, device=t.device),
            model_kwargs=model_kwargs,
            pred_xstart=None,
            lr_xt=lr_xt,
            coef_xt_reg=coef_xt_reg,
            conf=conf,
        )
        return result["x"], result["crop"]

