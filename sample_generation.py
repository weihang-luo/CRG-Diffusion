import os
import argparse
import re
import torch as th
from PIL import Image
import numpy as np
import conf_mgt
from utils import yamlread
from pathlib import Path
import json
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

from utils import save_grid, save_image, normalize_image
from utils.config import Config
from utils.logger import get_logger, logging_info
from utils.nn_utils import get_all_paths, set_random_seed
from utils.result_recorder import ResultRecorder
from utils.timer import Timer

from guided_diffusion import A_DDIMSampler
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
)

def load_and_preprocess_image(image_path, image_size=256):
    """Load and preprocess image"""
    pil_img = Image.open(image_path).convert('RGB')
    
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), Image.BICUBIC)
    
    arr_img = np.asarray(pil_img)
    img_tensor = th.tensor(arr_img).permute(2, 0, 1).float() / 127.5 - 1
    
    return img_tensor.unsqueeze(0)

def prepare_model(algorithm, conf, num_class, device):
    """Prepare model"""
    logging_info("Preparing model...")
    # Create main model
    unet = create_model(**select_args(conf, model_defaults().keys()))
    # Create crop region model
    crop_model = create_model(
        **conf['crop'], num_classes=num_class,
    )
    
    # Load sampler
    sampler_cls = A_DDIMSampler

    logging_info(f'Creating diffusion model [{sampler_cls.__name__}]...')
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    # Load model weights
    logging_info(f"Loading model from {conf.model_path}...")
    unet.load_state_dict(th.load(conf.model_path, weights_only=True, map_location=device))
    logging_info(f"Loading crop model from {conf.crop_model_path}...")
    crop_model.load_state_dict(th.load(conf.crop_model_path, weights_only=True, map_location=device))
    unet.to(device)
    crop_model.to(device)
    
    if conf.use_fp16:
        unet.convert_to_fp16()
        crop_model.convert_to_fp16()
    unet.eval()
    crop_model.eval()

    return unet, sampler, crop_model

def extract_info_from_filename(filename, defect_class_dict):
    """Extract defect type and location from filename"""
    basename = os.path.basename(filename)
    name_parts = os.path.splitext(basename)[0].split('_')
    
    cx, cy = None, None
    cx_idx, cy_idx = -1, -1
    defect_type_name = None
    
    # Find cx and cy positions
    for i, part in enumerate(name_parts):
        if part.startswith('cx'):
            try:
                cx = int(re.search(r'cx(\d+)', part).group(1))
                cx_idx = i
            except (AttributeError, ValueError):
                pass
        elif part.startswith('cy'):
            try:
                cy = int(re.search(r'cy(\d+)', part).group(1))
                cy_idx = i
            except (AttributeError, ValueError):
                pass
    
    # Extract defect type before coordinates
    if cx_idx > 0 and cy_idx > 0:
        if abs(cx_idx - cy_idx) == 1:
            defect_idx = min(cx_idx, cy_idx) - 1
            if defect_idx >= 0:
                defect_type_name = name_parts[defect_idx]
        else:
            if cx_idx > 0:
                defect_type_name = name_parts[cx_idx-1]
    
    if defect_type_name is None:
        defect_type_name = name_parts[1] if len(name_parts) > 1 else "defect0"
    
    defect_type_id = defect_class_dict.get(defect_type_name, 0)
    
    return defect_type_id, cx, cy, defect_type_name

def get_defect_positions_from_txt(txt_path, image_size, defect_class):
    """Read defect annotation from txt file"""
    defects = []
    
    if not os.path.exists(txt_path):
        logging_info(f"Warning: txt file not found - {txt_path}")
        return defects
    
    # Read coordinates from txt file
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    defect_type_id = int(parts[0])
                    # Convert relative coordinates to absolute coordinates
                    x_rel = float(parts[1])
                    y_rel = float(parts[2])
                    x = int(x_rel * image_size)
                    y = int(y_rel * image_size)
                    
                    defects.append([defect_type_id, x, y])
                except ValueError:
                    logging_info(f"Warning: invalid label format - {line}")
    
    if not defects:
        logging_info(f"Warning: no valid coordinates found in {txt_path}")
    
    return defects

def load_defect_class(conf):
    """Load defect classes"""
    if conf.crop_class_json_path:
        try:
            with Path(conf.crop_class_json_path).open('r') as f:
                defect_class = json.load(f)
                defect_count = len(defect_class)
                print(f"Loaded {defect_count} defect types from {conf.crop_class_json_path}")
                return defect_class, defect_count
        except Exception as e:
            print(f"Error loading defect class file: {e}")
    
    # Default defect type
    print("Using default defect types")
    return {"defect0": 0}, 1

def find_images_to_process(conf, defect_class):
    """Find images to process"""
    defect_free_images = []
    defect_free_dir = conf.defect_free_dir
    
    # Get output directory path
    all_paths = get_all_paths(os.path.join(conf.outdir))
    samples_dir = all_paths["path_sample"]
    
    print(f"Input defect-free image directory: {defect_free_dir}")
    print(f"Output generated image directory: {samples_dir}")
    
    # Check if directory exists
    if not os.path.isdir(defect_free_dir):
        print(f"Directory not found: {defect_free_dir}")
        return [], None
    
    # Analyze generated images
    image_status = {}
    if os.path.isdir(samples_dir):
        existing_samples = os.listdir(samples_dir)
        print(f"Found {len(existing_samples)} generated images")
        
        # Create image name mapping
        image_name_mapping = {}
        for filename in os.listdir(defect_free_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(filename)[0]
                image_name_mapping[base_name] = base_name
        
        # Analyze generated images
        for sample in existing_samples:
            if not sample.endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            parts = sample.split('_')
            if len(parts) < 4:
                continue
                
            # Find original image name
            original_name = None
            for name in image_name_mapping.keys():
                if sample.startswith(name + "_"):
                    original_name = name
                    break
            
            if original_name is None:
                original_name = parts[0]
            
            # Extract defect type and coordinates
            defect_type = None
            cx, cy = None, None
            
            for part in parts:
                if part in defect_class or part.startswith("defect"):
                    defect_type = part
                elif part.startswith('cx'):
                    try:
                        cx = int(re.search(r'cx(\d+)', part).group(1))
                    except (AttributeError, ValueError):
                        pass
                elif part.startswith('cy'):
                    try:
                        cy = int(re.search(r'cy(\d+)', part).group(1))
                    except (AttributeError, ValueError):
                        pass
            
            # Record valid combinations
            if defect_type and cx is not None and cy is not None:
                if original_name not in image_status:
                    image_status[original_name] = {
                        "defect_types": set(),
                        "total_files": 0,
                        "processed_combinations": set()
                    }
                
                image_status[original_name]["defect_types"].add(defect_type)
                image_status[original_name]["processed_combinations"].add((defect_type, cx, cy))
                image_status[original_name]["total_files"] += 1
    else:
        print(f"Output directory {samples_dir} does not exist, will create")
        os.makedirs(samples_dir, exist_ok=True)
    
    # Collect images to process
    for filename in sorted(os.listdir(defect_free_dir)):
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        base_name = os.path.splitext(filename)[0]
        original_name = base_name
        img_path = os.path.join(defect_free_dir, filename)
        
        # Check txt file existence
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        if not os.path.exists(txt_path):
            print(f"Skip image {filename}, no txt file found")
            continue
            
        # Check if already processed
        is_fully_processed = False
        if original_name in image_status:
            processed_combinations = image_status[original_name]["processed_combinations"]
            
            # Get expected combinations
            expected_combinations = []
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            defect_type_id = int(parts[0])
                            # Find defect type name
                            defect_type_name = None
                            for name, id_val in defect_class.items():
                                if id_val == defect_type_id:
                                    defect_type_name = name
                                    break
                            
                            if defect_type_name is None:
                                defect_type_name = f"defect{defect_type_id}"
                            
                            x_rel = float(parts[1])
                            y_rel = float(parts[2])
                            x = int(x_rel * conf.image_size)
                            y = int(y_rel * conf.image_size)
                            
                            expected_combinations.append((defect_type_name, x, y))
                        except ValueError:
                            print(f"Warning: invalid label format - {line}")
            
            # Check for unprocessed combinations
            unprocessed_combinations = [comb for comb in expected_combinations 
                                      if comb not in processed_combinations]
            
            if not unprocessed_combinations:
                is_fully_processed = True
                print(f"Skip processed image: {filename} (processed: {len(processed_combinations)})")
            else:
                print(f"Image {filename} has {len(unprocessed_combinations)}/{len(expected_combinations)} defect positions to process")
        
        # Add unprocessed images
        if not is_fully_processed:
            defect_free_images.append(img_path)
    
    print(f"Found {len(defect_free_images)} defect-free images to process")
    return defect_free_images, image_status

def process_images(conf, defect_free_images, image_status=None, device=None):
    """Process images on specified GPU"""
    batch_size = 1
    
    # Setup device
    if device is None:
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    # Prepare output directories and logger
    all_paths = get_all_paths(os.path.join(conf.outdir))
    conf.update(all_paths)
    get_logger(all_paths["path_log"], force_add_handler=True)
    
    # Initialize recorder
    recorder = ResultRecorder(
        path_record=all_paths["path_record"],
        initial_record=conf,
        use_git=conf.use_git
    )
    
    # Prepare output directories
    sample_path = all_paths["path_sample"]
    grid_path = all_paths["path_grid"]
    crop_path = all_paths["path_crop"]
    temp_path = all_paths["path_temp"]
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
    os.makedirs(crop_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)

    # Set random seed
    if conf.seed:
        set_random_seed(conf.seed, deterministic=False, no_torch=False, no_tf=True)
    else:
        np.random.seed(None)

    config = Config(default_config_dict=conf, use_argparse=False)
    config.attention_save_path = temp_path
    
    # Load defect classes
    if conf.crop_class_json_path:
        with Path(conf.crop_class_json_path).open('r') as f:
            defect_class = json.load(f)
    else:
        defect_class = {"defect0": 0}
    
    # Get class ID to name mapping
    class_id_to_name = {v: k for k, v in defect_class.items()}

    # Prepare device and model
    logging_info(f"Using device: {device}")
    unet, sampler, guide_models = prepare_model("fj_ddim", config, len(defect_class), device)
    
    # Define model function
    def model_fn(x, t, y=None, gt=None, **kwargs):
        return unet(x, t, y if conf.class_cond else None, gt=gt)

    # Defect region size
    crop_size = 64
    
    # Process all images
    logging_info(f"Start processing {len(defect_free_images)} images")
    
    for i, defect_img_path in enumerate(defect_free_images):
        logging_info(f"Processing image {i+1}/{len(defect_free_images)}: {defect_img_path}")
        
        # Get image base name
        image_base_name = os.path.splitext(os.path.basename(defect_img_path))[0]
        original_image_name = image_base_name
        logging_info(f"Original image name: {original_image_name}")
        
        # Get corresponding txt label file path
        txt_path = os.path.splitext(defect_img_path)[0] + '.txt'
        
        # Get processed defect type and coordinate combinations
        processed_combinations = set()
        if image_status and original_image_name in image_status:
            processed_combinations = image_status[original_image_name]["processed_combinations"]
            logging_info(f"Got {len(processed_combinations)} processed combinations from index")
        
        # Get defect positions
        defects = get_defect_positions_from_txt(txt_path, conf.image_size, defect_class)
        
        if not defects:
            logging_info(f"Warning: no valid defect labels found, skip processing")
            continue
            
        # Filter unprocessed defects
        defects_to_process = []
        for defect in defects:
            defect_type_id, cx, cy = defect
            defect_type_name = class_id_to_name.get(defect_type_id, f"defect{defect_type_id}")
            
            if (defect_type_name, cx, cy) in processed_combinations:
                logging_info(f"Skip processed defect: type={defect_type_name}({defect_type_id}), position=({cx},{cy})")
                continue
            
            defects_to_process.append(defect)
        
        if not defects_to_process:
            logging_info(f"Image {image_base_name} all defect combinations processed, skip")
            continue
            
        logging_info(f"Processing {len(defects_to_process)}/{len(defects)} unprocessed defect labels")
        
        # Load original defect-free image
        img = load_and_preprocess_image(defect_img_path, image_size=conf.image_size)
        img = img.to(device)
        
        # Process each defect label
        for defect_idx, defect in enumerate(defects_to_process):
            defect_type_id, cx, cy = defect
            defect_type_name = class_id_to_name.get(defect_type_id, f"defect{defect_type_id}")
            
            logging_info(f"Processing defect {defect_idx+1}/{len(defects_to_process)}: type={defect_type_name}({defect_type_id}), position=({cx},{cy})")
            
            # Batch copy image
            img_batch = img.repeat(batch_size, 1, 1, 1).to(device)
            
            # Ensure crop region doesn't exceed image boundary
            lx = max(0, min(cx - crop_size // 2, conf.image_size - crop_size))
            ly = max(0, min(cy - crop_size // 2, conf.image_size - crop_size))
            logging_info(f"Defect position: x={lx}, y={ly}, crop region=({lx},{ly})~({lx+crop_size},{ly+crop_size})")
            
            # Create unique image name with defect type and coordinates
            image_name = f"{image_base_name}_{defect_type_name}_cx{cx}_cy{cy}"
            
            # Setup model parameters
            model_kwargs = {}
            model_kwargs["img_free"] = img_batch
            model_kwargs["img"] = img_batch.clone()
            model_kwargs["crop_y"] = th.full((batch_size,), defect_type_id, device=device)
            model_kwargs["location"] = [lx, ly]
            
            # Add noise to image using forward process
            noise_step = conf["ddim"]["schedule_params"]["start_step"]
            logging_info(f"Adding noise to image at step {noise_step}...")

            noised_img = sampler.q_sample(img_batch, th.tensor([noise_step] * batch_size, device=device))

            # Get crop region from noised image
            crop_t = noised_img[:, :, ly:ly+crop_size, lx:lx+crop_size].clone()
            
            # Update model_kwargs
            model_kwargs["crop_t"] = crop_t
            model_kwargs["noise_step"] = noise_step
            model_kwargs["image_base_name"] = image_name
            
            # Start denoising from specified step
            logging_info(f"Start denoising from step {noise_step}...")
            
            timer = Timer()
            timer.start()
            
            # Call sampling loop
            result = sampler.p_sample_loop(
                model_fn,
                shape=(batch_size, 3, conf.image_size, conf.image_size),
                guide_models=guide_models,
                noise=noised_img,
                model_kwargs=model_kwargs,
                device=device,
                conf=config,
                start_step=noise_step
            )
            
            timer.end()
            
            logging_info(f"Saving generated result: {image_name}")
            
            # Preprocess images to save
            inpainted = normalize_image(result["sample"].detach().cpu())
            crop_result = normalize_image(result["crop"].detach().cpu())
            orig_img = normalize_image(img_batch.detach().cpu())
            noised_img_norm = normalize_image(noised_img.detach().cpu())
            
            # Save generated images
            for b in range(batch_size):
                # Save full image
                save_image(inpainted[b], os.path.join(sample_path, f"{image_name}_{b+1}.png"))
                # Save crop region
                save_image(crop_result[b], os.path.join(crop_path, f"{image_name}_crop_{b+1}.png"))
            
            # Save comparison grid
            samples = th.cat([orig_img, noised_img_norm, inpainted])
            save_grid(
                samples,
                os.path.join(grid_path, f"{image_name}_grid.png"),
                nrow=batch_size,
            )
            
            logging_info(f"Defect {defect_idx+1}/{len(defects_to_process)} completed, time: {timer.get_last_duration():.2f}s")
        
        logging_info(f"Image {i+1}/{len(defect_free_images)} all {len(defects_to_process)} defects processed")

    recorder.end_recording()
    logging_info("All images processed")


def main(conf, gpu_id=0):
    """Main function"""
    # Check GPU availability
    if th.cuda.is_available():
        available_device_count = th.cuda.device_count()
        print(f"System detected {available_device_count} available GPU(s)")
        
        if gpu_id >= available_device_count:
            print(f"Specified GPU ID {gpu_id} not available, using GPU 0")
            gpu_id = 0
            
        print(f"Using GPU ID: {gpu_id}")
        device = th.device(f"cuda:{gpu_id}")
    else:
        print("No GPU detected, using CPU mode")
        device = th.device("cpu")
    
    # Load defect classes
    defect_class, _ = load_defect_class(conf)
    
    # Find images to process
    defect_free_images, image_status = find_images_to_process(conf, defect_class)
    
    if len(defect_free_images) == 0:
        print("No images to process, exiting")
        return
    
    # Process all images
    process_images(conf, defect_free_images, image_status, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default="confs/pcb_gen.yml", help="Config file path")
    args = parser.parse_args()

    conf_arg = conf_mgt.conf_base.Default_Conf()
    if args.conf_path:
        conf_arg.update(yamlread(args.conf_path))
    
    main(conf_arg, conf_arg.cuda)

