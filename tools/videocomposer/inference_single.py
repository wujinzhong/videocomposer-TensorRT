import os
import os.path as osp
import sys
# append parent path to environment
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import logging
import numpy as np
# import copy
from copy import deepcopy, copy
import random
import json
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import torchvision.transforms as T
import pynvml
import torchvision.transforms.functional as TF
from importlib import reload
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
import open_clip
from easydict import EasyDict
from collections import defaultdict
from functools import partial
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.oss import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from .datasets import VideoDataset
import artist.ops as ops
import artist.data as data

from artist import DOWNLOAD_TO_CACHE
from artist.models.clip import VisionTransformer
import artist.models as models
from .config import cfg
from .unet_sd import UNetSD_temporal
from einops import rearrange
from artist.optim import Adafactor, AnnealingLR
from .autoencoder import  AutoencoderKL, DiagonalGaussianDistribution
from tools.annotator.canny import CannyDetector
from tools.annotator.sketch import pidinet_bsd, sketch_simplification_gan
# from tools.annotator.histogram import Palette
from utils.config import Config
import nvtx
import warnings
import trt_util_2
from trt_util_2 import (Memory_Manager, 
                        check_onnx,
                        TRT_Engine)

MODEL_TO_FLOAT16_ENABLED = True
 

def beta_schedule(schedule, num_timesteps=1000, init_beta=None, last_beta=None):
    '''
    This code defines a function beta_schedule that generates a sequence of beta values based on the given input parameters. These beta values can be used in video diffusion processes. The function has the following parameters:
        schedule(str): Determines the type of beta schedule to be generated. It can be 'linear', 'linear_sd', 'quadratic', or 'cosine'.
        num_timesteps(int, optional): The number of timesteps for the generated beta schedule. Default is 1000.
        init_beta(float, optional): The initial beta value. If not provided, a default value is used based on the chosen schedule.
        last_beta(float, optional): The final beta value. If not provided, a default value is used based on the chosen schedule.
    The function returns a PyTorch tensor containing the generated beta values. The beta schedule is determined by the schedule parameter:
        1.Linear: Generates a linear sequence of beta values betweeninit_betaandlast_beta.
        2.Linear_sd: Generates a linear sequence of beta values between the square root of init_beta and the square root oflast_beta, and then squares the result.
        3.Quadratic: Similar to the 'linear_sd' schedule, but with different default values forinit_betaandlast_beta.
        4.Cosine: Generates a sequence of beta values based on a cosine function, ensuring the values are between 0 and 0.999.
    If an unsupported schedule is provided, a ValueError is raised with a message indicating the issue.
    '''
    if schedule == 'linear':
        scale = 1000.0 / num_timesteps
        init_beta = init_beta or scale * 0.0001
        last_beta = last_beta or scale * 0.02
        return torch.linspace(init_beta, last_beta, num_timesteps, dtype=torch.float64)
    elif schedule == 'linear_sd':
        return torch.linspace(init_beta ** 0.5, last_beta ** 0.5, num_timesteps, dtype=torch.float64) ** 2
    elif schedule == 'quadratic':
        init_beta = init_beta or 0.0015
        last_beta = last_beta or 0.0195
        return torch.linspace(init_beta ** 0.5, last_beta ** 0.5, num_timesteps, dtype=torch.float64) ** 2
    elif schedule == 'cosine':
        betas = []
        for step in range(num_timesteps):
            t1 = step / num_timesteps
            t2 = (step + 1) / num_timesteps
            fn = lambda u: math.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
            betas.append(min(1.0 - fn(t2) / fn(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float64)
    else:
        raise ValueError(f'Unsupported schedule: {schedule}')


class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", pretrained="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        # 
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
        # 
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

def load_stable_diffusion_pretrained(state_dict, temporal_attention):
    import collections
    sd_new = collections.OrderedDict()
    keys = list(state_dict.keys())

    # "input_blocks.3.op.weight", "input_blocks.3.op.bias", "input_blocks.6.op.weight", "input_blocks.6.op.bias", "input_blocks.9.op.weight", "input_blocks.9.op.bias". 
    # "input_blocks.3.0.op.weight", "input_blocks.3.0.op.bias", "input_blocks.6.0.op.weight", "input_blocks.6.0.op.bias", "input_blocks.9.0.op.weight", "input_blocks.9.0.op.bias".
    for k in keys:
        if k.find('diffusion_model') >= 0:
            k_new = k.split('diffusion_model.')[-1]
            if k_new in ["input_blocks.3.0.op.weight", "input_blocks.3.0.op.bias", "input_blocks.6.0.op.weight", "input_blocks.6.0.op.bias", "input_blocks.9.0.op.weight", "input_blocks.9.0.op.bias"]:
                k_new = k_new.replace('0.op','op')
            if temporal_attention:
                if k_new.find('middle_block.2') >= 0:
                    k_new = k_new.replace('middle_block.2','middle_block.3')
                if k_new.find('output_blocks.5.2') >= 0:
                    k_new = k_new.replace('output_blocks.5.2','output_blocks.5.3')
                if k_new.find('output_blocks.8.2') >= 0:
                    k_new = k_new.replace('output_blocks.8.2','output_blocks.8.3')
            sd_new[k_new] = state_dict[k]

    return sd_new

def random_resize(img, size):
    img = [TF.resize(u, size, interpolation=random.choice([
        InterpolationMode.BILINEAR,
        InterpolationMode.BICUBIC,
        InterpolationMode.LANCZOS])) for u in img]
    return img

class CenterCrop(object):

    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        # fast resize
        while min(img.size) >= 2 * self.size:
            img = img.resize((img.width // 2, img.height // 2), resample=Image.BOX)
        scale = self.size / min(img.size)
        img = img.resize((round(scale * img.width), round(scale * img.height)), resample=Image.BICUBIC)

        # center crop
        x1 = (img.width - self.size) // 2
        y1 = (img.height - self.size) // 2
        img = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        return img

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        out = img + self.std * torch.randn_like(img) + self.mean        
        if out.dtype != dtype:
            out = out.to(dtype)
        return out
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def make_masked_images(imgs, masks):
    masked_imgs = []
    for i, mask in enumerate(masks):        
        # concatenation
        im = imgs[i] * (1 - mask)
        print(f"im: {im.device, im.shape, im.dtype}")
        masked_imgs.append(torch.cat([im, (1 - mask)], dim=1))
    return torch.stack(masked_imgs, dim=0)

@torch.no_grad()
def get_first_stage_encoding(encoder_posterior):
    scale_factor = 0.18215                                                                     
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * z


class FrozenOpenCLIPVisualEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", pretrained="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last", input_shape=(224, 224, 3)):
        super().__init__()
        assert layer in self.LAYERS
        # version = 'cache/open_clip_pytorch_model.bin'
        model, _, preprocess = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained) # '/mnt/workspace/videocomposer/VideoComposer_diffusion/cache/open_clip_pytorch_model.bin'
        # model, _, _ = open_clip.create_model_and_transforms(arch, device=device, pretrained=version)
        del model.transformer 
        self.model = model
        data_white=np.ones(input_shape, dtype=np.uint8) * 255
        self.black_image = preprocess(T.ToPILImage()(data_white)).unsqueeze(0)
        self.preprocess = preprocess

        self.device = device
        self.max_length = max_length # 77
        if freeze:
            self.freeze()
        self.layer = layer # 'penultimate'
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self): 
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        # tokens = open_clip.tokenize(text)
        if MODEL_TO_FLOAT16_ENABLED: image = image.to(torch.float16)
        z = self.model.encode_image(image.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


'''
{'__name__': 'Config: VideoComposer', 'video_compositions': ['text', 'mask', 'depthmap', 'sketch      ', 'motion', 'image', 'local_image', 'single_sketch'], 'root_dir': 'webvid10m/', 'alpha': 0.7, 'misc_size': 384, 'depth_std': 20      .0, 'depth_clamp': 10.0, 'hist_sigma': 10.0, 'use_image_dataset': False, 'alpha_img': 0.7, 'resolution': 256, 'mean': [0.5, 0.5,       0.5], 'std': [0.5, 0.5, 0.5], 'sketch_mean': [0.485, 0.456, 0.406], 'sketch_std': [0.229, 0.224, 0.225], 'max_words': 1000, 'fr      ame_lens': [16, 16, 16, 16], 'feature_framerates': [4], 'feature_framerate': 4, 'batch_sizes': {'1': 1, '4': 1, '8': 1, '16': 1}      , 'chunk_size': 1, 'num_workers': 0, 'prefetch_factor': 2, 'seed': 9999, 'num_timesteps': 1000, 'mean_type': 'eps', 'var_type':       'fixed_small', 'loss_type': 'mse', 'ddim_timesteps': 50, 'ddim_eta': 0.0, 'clamp': 1.0, 'share_noise': False, 'use_div_loss': Fa      lse, 'p_zero': 0.9, 'guide_scale': 6.0, 'sd_checkpoint': 'v2-1_512-ema-pruned.ckpt', 'vit_image_size': 224, 'vit_patch_size': 14      , 'vit_dim': 1024, 'vit_out_dim': 768, 'vit_heads': 16, 'vit_layers': 24, 'vit_mean': [0.48145466, 0.4578275, 0.40821073], 'vit_      std': [0.26862954, 0.26130258, 0.27577711], 'clip_checkpoint': 'open_clip_pytorch_model.bin', 'mvs_visual': False, 'unet_in_dim'      : 4, 'unet_concat_dim': 8, 'unet_y_dim': 768, 'unet_context_dim': 1024, 'unet_out_dim': 4, 'unet_dim': 320, 'unet_dim_mult': [1,       2, 4, 4], 'unet_res_blocks': 2, 'unet_num_heads': 8, 'unet_head_dim': 64, 'unet_attn_scales': [1.0, 0.5, 0.25], 'unet_dropout':       0.1, 'misc_dropout': 0.5, 'p_all_zero': 0.1, 'p_all_keep': 0.1, 'temporal_conv': False, 'temporal_attn_times': 1, 'temporal_att      ention': True, 'use_fps_condition': False, 'use_sim_mask': False, 'pretrained': False, 'fix_weight': False, 'resume': True, 'res      ume_step': 228000, 'resume_check_dir': '.', 'resume_checkpoint': 'model_weights/non_ema_228000.pth', 'resume_optimizer': False,       'use_ema': True, 'load_from': None, 'use_checkpoint': True, 'use_sharded_ddp': False, 'use_fsdp': False, 'use_fp16': True, 'ema_      decay': 0.9999, 'viz_interval': 1000, 'save_ckp_interval': 1000, 'log_interval': 100, 'log_dir': 'outputs/exp02_motion_transfer-      S09999', 'ENABLE': True, 'DATASET': 'webvid10m', 'TASK_TYPE': 'SINGLE_TASK', 'read_image': True, 'guidances': ['y', 'local_image      ', 'motion'], 'network_name': 'UNetSD_temporal', 'num_steps': 1, 'cfg_file': 'configs/exp02_motion_transfer.yaml', 'init_method'      : 'tcp://localhost:9999', 'debug': False, 'input_video': 'demo_video/motion_transfer.mp4', 'image_path': 'demo_video/moon_on_wat      er.jpg', 'sketch_path': '', 'style_image': None, 'input_text_desc': 'A beautiful big moon on the water at night', 'opts': [], 'r      ead_sketch': False, 'read_style': False, 'save_origin_video': True, 'pmi_rank': 0, 'pmi_world_size': 1, 'gpus_per_machine': 2, '      world_size': 2, 'gpu': 0, 'rank': 0, 'log_file': 'outputs/exp02_motion_transfer-S09999/exp02_motion_transfer-S09999_rank0.log'}
'''
def inference_single(cfg_update, **kwargs):
    cfg.update(**kwargs)

    # Copy update input parameter to current task
    for k, v in cfg_update.items():
        cfg[k] = v

    cfg.read_image = getattr(cfg, 'read_image', False)
    cfg.read_sketch = getattr(cfg, 'read_sketch', False)
    cfg.read_style = getattr(cfg, 'read_style', False)
    cfg.save_origin_video = getattr(cfg, 'save_origin_video', True)

    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) # 0
    print(f"pmi_rank: {cfg.pmi_rank}")
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    print(f"pmi_world_size: {cfg.pmi_world_size}")
    setup_seed(cfg.seed)

    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        print(f"gpus_per_machine: {cfg.gpus_per_machine}")
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
        print(f"world_size: {cfg.world_size}")
    
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg

def worker(gpu, cfg):
    
    GPU_device_str = ["6000Ada", "A6000"]
    GPU_device_str_idx = 0

    use_cleaner_fp16_trt_engine = True
    __cleaner_trt_model_path = "./onnx/cleaner_{}_{}.engine".format("fp16" if use_cleaner_fp16_trt_engine else "fp32", GPU_device_str[GPU_device_str_idx])
    __cleaner_trt_engine = None

    use_pidinet_fp16_trt_engine = True
    __pidinet_trt_model_path = "./onnx/pidinet_{}_{}.engine".format("fp16" if use_pidinet_fp16_trt_engine else "fp32", GPU_device_str[GPU_device_str_idx])
    __pidinet_trt_engine = None

    use_midas_fp16_trt_engine = True
    __midas_trt_model_path = "./onnx/midas_{}_{}.engine".format("fp16" if use_midas_fp16_trt_engine else "fp32", GPU_device_str[GPU_device_str_idx])
    __midas_trt_engine = None

    use_autoencoder_encode_fp16_trt_engine = True
    __autoencoder_encode_trt_model_path = "./onnx/autoencoder_encode_{}_{}.engine".format("fp16" if use_autoencoder_encode_fp16_trt_engine else "fp32", GPU_device_str[GPU_device_str_idx])
    __autoencoder_encode_trt_engine = None

    use_autoencoder_decode_fp16_trt_engine = True
    __autoencoder_decode_trt_model_path = "./onnx/autoencoder_decode_{}_{}.engine".format("fp16" if use_autoencoder_decode_fp16_trt_engine else "fp32", GPU_device_str[GPU_device_str_idx])
    __autoencoder_decode_trt_engine = None

    memory_manager=Memory_Manager()
    memory_manager.add_foot_print("before init")
    
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    cur_torch_device = torch.device(f'cuda:{gpu}')
    print(f"rank: {cfg.rank}")

    # init distributed processes
    torch.cuda.set_device(gpu)
    torch_stream = torch.cuda.Stream()
    if torch_stream is not None: torch.cuda.set_stream(torch_stream)
    cfg_mean_gpu=torch.tensor(cfg.mean,device=torch.device(f"cuda:{gpu}")).view(1, -1, 1, 1, 1)#ncfhw
    cfg_std_gpu=torch.tensor(cfg.std,device=torch.device(f"cuda:{gpu}")).view(1, -1, 1, 1, 1)#ncfhw

    torch.backends.cudnn.benchmark = True
    if not cfg.debug:
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # logging
    log_dir = ops.generalized_all_gather(cfg.log_dir)[0]
    exp_name = os.path.basename(cfg.cfg_file).split('.')[0] + '-S%05d' % (cfg.seed)
    log_dir = os.path.join(log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir
    if cfg.rank == 0:
        name = osp.basename(cfg.log_dir)
        cfg.log_file = osp.join(cfg.log_dir, '{}_rank{}.log'.format(name, cfg.rank))
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=cfg.log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)

    # rank-wise params
    l1 = len(cfg.frame_lens)
    l2 = len(cfg.feature_framerates)
    cfg.max_frames = cfg.frame_lens[cfg.rank % (l1*l2)// l2]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]
    
    rng = nvtx.start_range(message="init", color="blue")
    # [Transformer] Transformers for different inputs
    infer_trans = data.Compose([
        data.CenterCropV2(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])

    misc_transforms = data.Compose([
        T.Lambda(partial(random_resize, size=cfg.misc_size)),
        data.CenterCropV2(cfg.misc_size),
        data.ToTensor()])

    mv_transforms = data.Compose([
        T.Resize(size=cfg.resolution),
        T.CenterCrop(cfg.resolution)])
    
    memory_manager.add_foot_print("misc")

    dataset = VideoDataset(
        cfg=cfg,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        transforms=infer_trans,
        mv_transforms=mv_transforms,
        misc_transforms=misc_transforms,
        vit_transforms=T.Compose([
            CenterCrop(cfg.vit_image_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)]),
        vit_image_size= cfg.vit_image_size,
        misc_size=cfg.misc_size)

    dataloader = DataLoader(
        dataset=dataset,
        num_workers=0,
        pin_memory=True)
    memory_manager.add_foot_print("dataset")

    clip_encoder = FrozenOpenCLIPEmbedder(layer='penultimate',pretrained = DOWNLOAD_TO_CACHE(cfg.clip_checkpoint))
    clip_encoder.model.to(gpu)
    if MODEL_TO_FLOAT16_ENABLED: clip_encoder.to(torch.float16)
    zero_y = clip_encoder("").detach() # [1, 77, 1024]
    
    clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(layer='penultimate',pretrained = DOWNLOAD_TO_CACHE(cfg.clip_checkpoint))
    clip_encoder_visual.model.to(gpu)
    if MODEL_TO_FLOAT16_ENABLED: clip_encoder_visual.model.to(torch.float16)
    if MODEL_TO_FLOAT16_ENABLED: clip_encoder_visual.black_image.to(torch.float16)
    black_image_feature = clip_encoder_visual(clip_encoder_visual.black_image).unsqueeze(1) # [1, 1, 1024]
    black_image_feature = torch.zeros_like(black_image_feature) # for old

    memory_manager.add_foot_print("clip model")

    frame_in = None
    if cfg.read_image:
        image_key = cfg.image_path # 
        frame = Image.open(open(image_key, mode='rb')).convert('RGB')
        frame_in = misc_transforms([frame]) 
    
    frame_sketch = None
    if cfg.read_sketch:
        sketch_key = cfg.sketch_path # 
        frame_sketch = Image.open(open(sketch_key, mode='rb')).convert('RGB')
        frame_sketch = misc_transforms([frame_sketch]) # 

    frame_style = None
    if cfg.read_style:
        frame_style = Image.open(open(cfg.style_image, mode='rb')).convert('RGB')
    trt_util_2.synchronize( torch_stream )
    memory_manager.add_foot_print("read images")
    nvtx.end_range(rng)
    rng = nvtx.start_range(message="generators", color="blue")
    # [Contions] Generators for various conditions
    if 'depthmap' in cfg.video_compositions:
        if False: #baseline, torch native fp16
            midas = models.midas_v3(pretrained=True).eval().requires_grad_(False).to(
                memory_format=torch.channels_last).half().to(gpu)
        else: #trt
            midas = models.midas_v3(pretrained=True).eval().requires_grad_(False).to(
                memory_format=torch.channels_last).to(gpu)
            torch_onnx_export_midas( midas, onnx_model_path="./onnx/midas.onnx" )
            print(f"__midas_trt_engine, begin creating TRT_Engine......")
            if __midas_trt_engine is None: 
                __midas_trt_engine = TRT_Engine(__midas_trt_model_path, gpu_id=gpu, torch_stream=torch_stream)
            assert __midas_trt_engine
            if __midas_trt_engine is None:
                return
            print(f"__midas_trt_engine, end creating TRT_Engine.")
    
    memory_manager.add_foot_print("depthmap model")
    if 'canny' in cfg.video_compositions:
        canny_detector = CannyDetector()
    memory_manager.add_foot_print("canny model")
    if 'sketch' in cfg.video_compositions:
        pidinet = pidinet_bsd(pretrained=True, vanilla_cnn=True).eval().requires_grad_(False).to(gpu)
        onnx_model_path="./onnx/pidinet.onnx"
        torch_onnx_export_pidinet( pidinet, onnx_model_path=onnx_model_path )
        if os.path.exists(onnx_model_path):# after build trt engine, open this block
            print(f"__pidinet_trt_engine, begin creating TRT_Engine......")
            if __pidinet_trt_engine is None: 
                __pidinet_trt_engine = TRT_Engine(__pidinet_trt_model_path, gpu_id=gpu, torch_stream=torch_stream)
            assert __pidinet_trt_engine
            if __pidinet_trt_engine is None:
                return
            print(f"__pidinet_trt_engine, end creating TRT_Engine.")
        
        cleaner = sketch_simplification_gan(pretrained=True).eval().requires_grad_(False).to(gpu)
        onnx_model_path="./onnx/cleaner.onnx"
        torch_onnx_export_cleaner( cleaner, onnx_model_path=onnx_model_path )
        if os.path.exists(onnx_model_path):# after build trt engine, open this block
            print(f"__cleaner_trt_engine, begin creating TRT_Engine......")
            if __cleaner_trt_engine is None: 
                __cleaner_trt_engine = TRT_Engine(__cleaner_trt_model_path, gpu_id=gpu, torch_stream=torch_stream)
            assert __cleaner_trt_engine
            if __cleaner_trt_engine is None:
                return
            print(f"__cleaner_trt_engine, end creating TRT_Engine.")

        pidi_mean = torch.tensor(cfg.sketch_mean).view(1, -1, 1, 1).to(gpu)
        pidi_std = torch.tensor(cfg.sketch_std).view(1, -1, 1, 1).to(gpu)

        if MODEL_TO_FLOAT16_ENABLED: pidinet.to(torch.float16)
        if MODEL_TO_FLOAT16_ENABLED: cleaner.to(torch.float16)
        if MODEL_TO_FLOAT16_ENABLED: pidi_mean.to(torch.float16)
        if MODEL_TO_FLOAT16_ENABLED: pidi_std.to(torch.float16)
    
    # Placeholder for color inference
    palette = None
    trt_util_2.synchronize( torch_stream )
    memory_manager.add_foot_print("sketch model")
    nvtx.end_range(rng)
    rng = nvtx.start_range(message="auotoencoder", color="blue")
    # [model] auotoencoder
    ddconfig = {'double_z': True, 'z_channels': 4, \
                'resolution': 256, 'in_channels': 3, \
                'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], \
                'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    autoencoder = AutoencoderKL(ddconfig, 4, ckpt_path=DOWNLOAD_TO_CACHE(cfg.sd_checkpoint))
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    #autoencoder.to(torch.float16)
    memory_manager.add_foot_print("autoencoder model")

    torch_onnx_export_autoencoder_decode( autoencoder.decoder, onnx_model_path="./onnx/autoencoder_decode.onnx" )
    torch_onnx_export_autoencoder_encode( autoencoder.encoder, onnx_model_path="./onnx/autoencoder_encode.onnx" )
    
    if __autoencoder_encode_trt_engine is None: 
        __autoencoder_encode_trt_engine = TRT_Engine(__autoencoder_encode_trt_model_path, gpu_id=gpu, torch_stream=torch_stream)
        assert __autoencoder_encode_trt_engine
        if __autoencoder_encode_trt_engine is None:
            return
    if __autoencoder_decode_trt_engine is None: 
        __autoencoder_decode_trt_engine = TRT_Engine(__autoencoder_decode_trt_model_path, gpu_id=gpu, torch_stream=torch_stream)
        assert __autoencoder_decode_trt_engine
        if __autoencoder_decode_trt_engine is None:
            return
        
    if hasattr(cfg, "network_name") and cfg.network_name == "UNetSD_temporal":
        model = UNetSD_temporal(
            cfg=cfg,
            in_dim=cfg.unet_in_dim,
            concat_dim= cfg.unet_concat_dim,
            dim=cfg.unet_dim,
            y_dim=cfg.unet_y_dim,
            context_dim=cfg.unet_context_dim,
            out_dim=cfg.unet_out_dim,
            dim_mult=cfg.unet_dim_mult,
            num_heads=cfg.unet_num_heads,
            head_dim=cfg.unet_head_dim,
            num_res_blocks=cfg.unet_res_blocks,
            attn_scales=cfg.unet_attn_scales,
            dropout=cfg.unet_dropout,
            temporal_attention = cfg.temporal_attention,
            temporal_attn_times = cfg.temporal_attn_times,
            use_checkpoint=cfg.use_checkpoint,
            use_fps_condition=cfg.use_fps_condition,
            use_sim_mask=cfg.use_sim_mask,
            video_compositions=cfg.video_compositions,
            misc_dropout=cfg.misc_dropout,
            p_all_zero=cfg.p_all_zero,
            p_all_keep=cfg.p_all_zero,
            zero_y = zero_y,
            black_image_feature = black_image_feature,
            ).to(gpu)
        if MODEL_TO_FLOAT16_ENABLED: model.to(torch.float16)
    else:
        logging.info("Other model type not implement, exist")
        raise NotImplementedError(f"The model {cfg.network_name} not implement")
        return 
    trt_util_2.synchronize( torch_stream )
    memory_manager.add_foot_print("UNet model")
    nvtx.end_range(rng)
    rng = nvtx.start_range(message="checkpoint", color="blue")
    # Load checkpoint
    resume_step = 1
    if cfg.resume and cfg.resume_checkpoint:
        if hasattr(cfg, "text_to_video_pretrain") and cfg.text_to_video_pretrain:
            ss = torch.load(DOWNLOAD_TO_CACHE(cfg.resume_checkpoint))
            ss = {key:p for key,p in ss.items() if 'input_blocks.0.0' not in key}
            model.load_state_dict(ss,strict=False)
        else:
            model.load_state_dict(torch.load(DOWNLOAD_TO_CACHE(cfg.resume_checkpoint), map_location='cpu'),strict=False)
        if cfg.resume_step:
            resume_step = cfg.resume_step
        
        logging.info(f'Successfully load step {resume_step} model from {cfg.resume_checkpoint}')
        torch.cuda.empty_cache()
    else:
        logging.error(f'The checkpoint file {cfg.resume_checkpoint} is wrong')
        raise ValueError(f'The checkpoint file {cfg.resume_checkpoint} is wrong ')
        return
    
    # mark model size
    if cfg.rank == 0:
        logging.info(f'Created a model with {int(sum(p.numel() for p in model.parameters()) / (1024 ** 2))}M parameters')
    trt_util_2.synchronize( torch_stream )
    memory_manager.add_foot_print("checkpoint")
    nvtx.end_range(rng)
    rng = nvtx.start_range(message="diffusion", color="blue")
    # diffusion
    betas = beta_schedule('linear_sd', cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120)
    diffusion = ops.GaussianDiffusion(
        betas=betas,
        mean_type=cfg.mean_type,
        var_type=cfg.var_type,
        loss_type=cfg.loss_type,
        rescale_timesteps=False)
    trt_util_2.synchronize( torch_stream )
    nvtx.end_range(rng)
    memory_manager.add_foot_print("difussion model")
    # global variables
    viz_num = cfg.batch_size
    model = model.eval()

    for I in range(3):
        if I==0:         rng_for = nvtx.start_range(message="warmup", color="blue")
        else:            rng_for = nvtx.start_range(message="for", color="blue")
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
            #if True:
                for step, batch in enumerate(dataloader):
                    rng = nvtx.start_range(message=f"iter{step}", color="red")
                    
                    caps = batch[1]; del batch[1]
                    batch = ops.to_device(batch, gpu, non_blocking=True)
                    if cfg.max_frames == 1 and cfg.use_image_dataset:
                        ref_imgs, video_data, misc_data, mask, mv_data = batch
                        fps =  torch.tensor([cfg.feature_framerate]*cfg.batch_size,dtype=torch.long, device=gpu)
                    else:
                        ref_imgs, video_data, misc_data, fps, mask, mv_data = batch

                    ### save for visualization
                    misc_backups = copy(misc_data)
                    misc_backups = rearrange(misc_backups, 'b f c h w -> b c f h w')
                    mv_data_video = []
                    if 'motion' in cfg.video_compositions:
                        mv_data_video = rearrange(mv_data, 'b f c h w -> b c f h w')
                    trt_util_2.synchronize( torch_stream )
                    nvtx.end_range(rng)
                    rng = nvtx.start_range(message="mask", color="red")
                    ### mask images
                    masked_video = []
                    if 'mask' in cfg.video_compositions:
                        masked_video = make_masked_images(misc_data.sub(0.5).div_(0.5), mask)
                        masked_video = rearrange(masked_video, 'b f c h w -> b c f h w')
                    trt_util_2.synchronize( torch_stream )
                    nvtx.end_range(rng)
                    rng = nvtx.start_range(message="local_image", color="red")
                    ### Single Image
                    image_local = []
                    if 'local_image' in cfg.video_compositions:
                        frames_num = misc_data.shape[1]
                        bs_vd_local = misc_data.shape[0]
                        if cfg.read_image:
                            image_local = frame_in.unsqueeze(0).repeat(bs_vd_local,frames_num,1,1,1).cuda()
                        else:
                            image_local = misc_data[:,:1].clone().repeat(1,frames_num,1,1,1)
                        image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
                    trt_util_2.synchronize( torch_stream )
                    nvtx.end_range(rng)
                    rng = nvtx.start_range(message="encode", color="red")
                    ### encode the video_data
                    bs_vd = video_data.shape[0]
                    video_data_origin = video_data.clone() 
                    video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
                    misc_data = rearrange(misc_data, 'b f c h w -> (b f) c h w')
                    # video_data_origin = video_data.clone() 

                    video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
                    misc_data_list = torch.chunk(misc_data, misc_data.shape[0]//cfg.chunk_size,dim=0)

                    with torch.no_grad():
                        decode_data = []
                        rng_autoencoder = nvtx.start_range(message="autoencoder()", color="red")
                        for vd_data in video_data_list:
                            #print(f"autoencoder.encode(), vd_data: {vd_data.device, vd_data.shape, vd_data.dtype}")
                            if __autoencoder_encode_trt_engine:
                                nvtx_trt = nvtx.start_range(message='trt_encode', color='red')
                                encoder_output = __autoencoder_encode_trt_engine.inference(inputs=[vd_data],
                                                                                           outputs = __autoencoder_encode_trt_engine.output_tensors)
                                assert encoder_output is None
                                moments = __autoencoder_encode_trt_engine.output_tensors[0].to(torch.float16)
                                trt_util_2.synchronize( torch_stream )
                                nvtx.end_range(nvtx_trt)
                            else:
                                moments = autoencoder.encode(vd_data)
                            #print(f"autoencoder.encode(), moments: {moments.device, moments.shape, moments.dtype}")
                            encoder_posterior = DiagonalGaussianDistribution(moments)
                            tmp = get_first_stage_encoding(encoder_posterior).detach()
                            decode_data.append(tmp)
                        trt_util_2.synchronize( torch_stream )
                        nvtx.end_range(rng_autoencoder)
                        video_data = torch.cat(decode_data,dim=0)
                        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = bs_vd)
                        
                        trt_util_2.synchronize( torch_stream )
                        nvtx.end_range(rng)
                        rng = nvtx.start_range(message="depthmap", color="red")
                        depth_data = []
                        if 'depthmap' in cfg.video_compositions:
                            for misc_imgs in misc_data_list:
                                midas_input = misc_imgs.sub(0.5).div_(0.5).to(memory_format=torch.channels_last).half()
                                
                                #--------------------------------
                                if __midas_trt_engine:
                                    trt_util_2.synchronize( torch_stream )
                                    nvtx_trt = nvtx.start_range(message='midas', color='red')
                                    #inputs/outpus of midas/trt are all fp32
                                    midas_output = __midas_trt_engine.inference(inputs=[midas_input.to(torch.float32)],
                                                                                            outputs = __midas_trt_engine.output_tensors)
                                    assert midas_output is None
                                    depth = __midas_trt_engine.output_tensors[0].to(torch.float16)
                                    trt_util_2.synchronize( torch_stream )
                                    nvtx.end_range(nvtx_trt)
                                else:
                                    depth = midas(midas_input)
                                #--------------------------------
                                
                                
                                depth = (depth / cfg.depth_std).clamp_(0, cfg.depth_clamp)
                                depth_data.append(depth)
                            depth_data = torch.cat(depth_data, dim = 0)
                            depth_data = rearrange(depth_data, '(b f) c h w -> b c f h w', b = bs_vd)
                        trt_util_2.synchronize( torch_stream )
                        nvtx.end_range(rng)
                        rng = nvtx.start_range(message="canny", color="red")
                        canny_data = []
                        if 'canny' in cfg.video_compositions:
                            for misc_imgs in misc_data_list:
                                # print(misc_imgs.shape)
                                misc_imgs = rearrange(misc_imgs.clone(), 'k c h w -> k h w c') # 'k' means 'chunk'.
                                canny_condition = torch.stack([canny_detector(misc_img) for misc_img in misc_imgs])
                                canny_condition = rearrange(canny_condition, 'k h w c-> k c h w')
                                canny_data.append(canny_condition)
                            canny_data = torch.cat(canny_data, dim = 0)
                            canny_data = rearrange(canny_data, '(b f) c h w -> b c f h w', b = bs_vd)
                        trt_util_2.synchronize( torch_stream )
                        nvtx.end_range(rng)
                        rng = nvtx.start_range(message="sketch", color="red")
                        sketch_data = []
                        if 'sketch' in cfg.video_compositions:
                            sketch_list = misc_data_list
                            if cfg.read_sketch:
                                sketch_repeat = frame_sketch.repeat(frames_num, 1, 1, 1).cuda()
                                sketch_list = [sketch_repeat]

                            for misc_imgs in sketch_list:
                                pidinet_input = misc_imgs.sub(pidi_mean).div_(pidi_std)
                                
                                trt_util_2.synchronize( torch_stream )
                                rng_ = nvtx.start_range(message="pidinet", color="red")

                                if __pidinet_trt_engine:
                                    trt_util_2.synchronize( torch_stream )
                                    #inputs/outpus of pidinet/trt are all fp32
                                    pidinet_output = __pidinet_trt_engine.inference(inputs=[pidinet_input.to(torch.float32)],
                                                                                            outputs = __pidinet_trt_engine.output_tensors)
                                    assert pidinet_output is None
                                    sketch = __pidinet_trt_engine.output_tensors[0].to(torch.float16)
                                    trt_util_2.synchronize( torch_stream )
                                else:
                                    sketch = pidinet(pidinet_input)
                                
                                trt_util_2.synchronize( torch_stream )
                                nvtx.end_range(rng_)
                                rng_ = nvtx.start_range(message="cleaner", color="red")
                                cleaner_input = 1.0 - sketch
                                
                                if __cleaner_trt_engine:
                                    trt_util_2.synchronize( torch_stream )
                                    #inputs/outpus of cleaner/trt are all fp32
                                    cleaner_output = __cleaner_trt_engine.inference(inputs=[cleaner_input.to(torch.float32)],
                                                                                            outputs = __cleaner_trt_engine.output_tensors)
                                    assert cleaner_output is None
                                    sketch = __cleaner_trt_engine.output_tensors[0].to(torch.float16)
                                    trt_util_2.synchronize( torch_stream )
                                else:
                                    sketch = 1.0 - cleaner(cleaner_input)
                                
                                trt_util_2.synchronize( torch_stream )
                                nvtx.end_range(rng_)

                                sketch_data.append(sketch)
                            sketch_data = torch.cat(sketch_data, dim = 0)
                            sketch_data = rearrange(sketch_data, '(b f) c h w -> b c f h w', b = bs_vd)
                        trt_util_2.synchronize( torch_stream )
                        nvtx.end_range(rng)
                        rng = nvtx.start_range(message="single_sketch", color="red")
                        single_sketch_data = []
                        if 'single_sketch' in cfg.video_compositions:
                            single_sketch_data = sketch_data.clone()[:, :, :1].repeat(1, 1, frames_num, 1, 1)
                        trt_util_2.synchronize( torch_stream )
                        nvtx.end_range(rng)

                    # preprocess for input text descripts
                    y = clip_encoder(caps).detach()  # [1, 77, 1024]
                    y0 = y.clone()
                    
                    rng = nvtx.start_range(message="clip", color="red")
                    y_visual = []
                    if 'image' in cfg.video_compositions:
                        with torch.no_grad():
                            if cfg.read_style:
                                y_visual = clip_encoder_visual(clip_encoder_visual.preprocess(frame_style).unsqueeze(0).cuda()).unsqueeze(0)
                                y_visual0 = y_visual.clone()
                            else:
                                ref_imgs = ref_imgs.squeeze(1)
                                y_visual = clip_encoder_visual(ref_imgs).unsqueeze(1) # [1, 1, 1024]
                                y_visual0 = y_visual.clone()
                    trt_util_2.synchronize( torch_stream )
                    nvtx.end_range(rng)
                    
                    with torch.no_grad():
                        # Log memory
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')
                        
                        rng_ddim = nvtx.start_range(message="DDIM", color="red")
                        # Sample images (DDIM)
                        with amp.autocast(enabled=cfg.use_fp16):
                            rng = nvtx.start_range(message="preDDIM", color="red")
                            if cfg.share_noise:
                                b, c, f, h, w = video_data.shape
                                noise = torch.randn((viz_num, c, h, w), device=gpu)
                                noise = noise.repeat_interleave(repeats=f, dim=0) 
                                noise = rearrange(noise, '(b f) c h w->b c f h w', b = viz_num) 
                                noise = noise.contiguous()
                            else:
                                noise=torch.randn_like(video_data[:viz_num])

                            full_model_kwargs=[
                                {'y': y0[:viz_num],
                                "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                                'image': None if len(y_visual) == 0 else y_visual0[:viz_num],
                                'depth': None if len(depth_data) == 0 else depth_data[:viz_num],
                                'canny': None if len(canny_data) == 0 else canny_data[:viz_num],
                                'sketch': None if len(sketch_data) == 0 else sketch_data[:viz_num],
                                'masked': None if len(masked_video) == 0 else masked_video[:viz_num],
                                'motion': None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                                'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                                'fps': fps[:viz_num]}, 
                                {'y': zero_y.repeat(viz_num,1,1) if not cfg.use_fps_condition else torch.zeros_like(y0)[:viz_num],
                                "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                                'image': None if len(y_visual) == 0 else torch.zeros_like(y_visual0[:viz_num]),
                                'depth': None if len(depth_data) == 0 else depth_data[:viz_num],
                                'canny': None if len(canny_data) == 0 else canny_data[:viz_num],
                                'sketch': None if len(sketch_data) == 0 else sketch_data[:viz_num],
                                'masked': None if len(masked_video) == 0 else masked_video[:viz_num],
                                'motion': None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                                'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                                'fps': fps[:viz_num]}
                            ]
                            # Save generated videos 
                            #--------------------------------------
                            partial_keys = cfg.guidances
                            noise_motion = noise.clone()
                            model_kwargs = prepare_model_kwargs(partial_keys = partial_keys,
                                                    full_model_kwargs = full_model_kwargs,
                                                    use_fps_condition = cfg.use_fps_condition)
                            trt_util_2.synchronize( torch_stream )
                            nvtx.end_range(rng)
                            rng = nvtx.start_range(message="ddim_sample_loop", color="red")
                            video_output = diffusion.ddim_sample_loop(
                                noise=noise_motion,
                                model=model, #model.eval(),
                                model_kwargs=model_kwargs,
                                guide_scale=9.0,
                                ddim_timesteps=cfg.ddim_timesteps,
                                eta=0.0,
                                torch_stream=torch_stream)
                            trt_util_2.synchronize( torch_stream )
                            nvtx.end_range(rng)
                            rng = nvtx.start_range(message="visualize_with_model_kwargs", color="red")
                            
                            visualize_with_model_kwargs(model_kwargs = model_kwargs,
                                video_data = video_output,
                                autoencoder = autoencoder,
                                ori_video = misc_backups,
                                viz_num = viz_num,
                                step = step,
                                caps = caps,
                                palette = palette,
                                cfg = cfg,
                                mean = cfg_mean_gpu,
                                std = cfg_std_gpu,
                                torch_stream = torch_stream,
                                autoencoder_decode_trt_engine = __autoencoder_decode_trt_engine,
                                autoencoder_decode_trt_engine_outputs = __autoencoder_decode_trt_engine.output_tensors if __autoencoder_decode_trt_engine else None,
                                testIdx = I )
                            trt_util_2.synchronize( torch_stream )
                            nvtx.end_range(rng)
                            #--------------------------------------
                        trt_util_2.synchronize( torch_stream )
                        nvtx.end_range(rng_ddim)
                    trt_util_2.synchronize( torch_stream )
                    torch.cuda.synchronize()
                    memory_manager.add_foot_print(f"for{step}")
        trt_util_2.synchronize( torch_stream )
        nvtx.end_range(rng_for)
    memory_manager.summary()

    if cfg.rank == 0:
        # send a sign to oss to indicate the training is completed
        logging.info('Congratulations! The inference is completed!')
    
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition):
    for partial_key in partial_keys:
        assert partial_key in ['y', 'depth', 'canny', 'masked', 'sketch', "image", "motion", "local_image", "single_sketch"]
    
    if use_fps_condition is True:
        partial_keys.append('fps')
    
    partial_model_kwargs = [{}, {}]
    for partial_key in partial_keys:
        partial_model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
        partial_model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]

    return partial_model_kwargs


def visualize_with_model_kwargs(model_kwargs,
                                video_data,
                                autoencoder,
                                # ref_imgs,
                                ori_video,
                                viz_num,
                                step,
                                caps,
                                palette,
                                cfg,
                                mean,
                                std,
                                torch_stream = None,
                                autoencoder_decode_trt_engine = None,
                                autoencoder_decode_trt_engine_outputs = [],
                                testIdx = 0,
                                ):
    scale_factor = 0.18215
    video_data = 1. / scale_factor * video_data
    #print(f"pos0, video_data: {video_data.device, video_data.shape, video_data.dtype}")
    
    bs_vd = video_data.shape[0]
    video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
    chunk_size = min(16, video_data.shape[0])
    video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
    decode_data = []

    trt_util_2.synchronize( torch_stream )
    rng = nvtx.start_range(message="autoencoder.decode", color="blue")
    for vd_data in video_data_list:
        #print(f"autoencoder.decode, vd_data: {vd_data.device, vd_data.shape, vd_data.dtype}") #vd_data: (device(type='cuda', index=0), torch.Size([1, 3, 256, 256]), torch.float32)
        
        if autoencoder_decode_trt_engine is not None:
            nvtx_trt = nvtx.start_range(message='trt_decode', color='red')
            unet_output = autoencoder_decode_trt_engine.inference(inputs=[vd_data],
                                            outputs = autoencoder_decode_trt_engine_outputs)
            assert unet_output is None
            tmp = autoencoder_decode_trt_engine_outputs[0].to(torch.float16)
            trt_util_2.synchronize( torch_stream )
            nvtx.end_range(nvtx_trt)
        else:
            tmp = autoencoder.decode(vd_data).to(torch.device('cuda'))
        #print(f"output autoencoder.decode, tmp: {tmp.device, tmp.shape, tmp.dtype}")
        
        decode_data.append(tmp)
    trt_util_2.synchronize( torch_stream )
    nvtx.end_range(rng)
    video_data = torch.cat(decode_data,dim=0)
    video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = bs_vd)
    #print(f"pos1, video_data: {video_data.device, video_data.shape, video_data.dtype}")
    ori_video = ori_video[:viz_num]
    
    oss_key = os.path.join(cfg.log_dir, f"rank_{cfg.world_size}-{cfg.rank}.gif")
    text_key = osp.join(cfg.log_dir, 'text_description.txt')
    
    # Save videos and text inputs.
    rng = nvtx.start_range(message="save_video", color="blue")
    try:
        del model_kwargs[0][list(model_kwargs[0].keys())[0]]
        del model_kwargs[1][list(model_kwargs[1].keys())[0]]
        if False:#imageio/CPU
            ops.save_video_multiple_conditions_imageio(
                oss_key, 
                video_data, 
                model_kwargs, 
                ori_video, 
                palette,
                cfg.mean, 
                cfg.std, 
                nrow=1, 
                save_origin_video=cfg.save_origin_video,
                torch_stream=torch_stream)
        else:#VPF
            ops.save_video_multiple_conditions_VPF(
                oss_key, 
                video_data, 
                model_kwargs, 
                ori_video, 
                palette,
                mean = mean, #cfg.mean, 
                std = std, 
                nrow=1, 
                save_origin_video=cfg.save_origin_video,
                torch_stream=torch_stream,
                testIdx=testIdx)
        if cfg.rank == 0: 
            texts = '\n'.join(caps[:viz_num])
            open(text_key, 'w').writelines(texts)
    except Exception as e:
        logging.info(f'Save text or video error. {e}')

    trt_util_2.synchronize( torch_stream )
    nvtx.end_range(rng)
    logging.info(f'Save videos to {oss_key}')

def torch_onnx_export_autoencoder_decode( model, onnx_model_path ):
    if not os.path.exists(onnx_model_path):
        #dynamic_axes = {
        #    "latent_model_input":   {0: "bs_x_2"},
        #    "prompt_embeds":        {0: "bs_x_2"},
        #    "noise_pred":           {0: "batch_size"}
        #}

        device = torch.device("cuda:0")
        dummy_inputs = {
            "input": torch.randn((16, 4, 32, 32),dtype=torch.float32).to(device),
        }
        output_names = ["posterior"]

        #import apex
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                #with open(onnx_model_path, "wb") as f:
                with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
                    torch.onnx.export(
                        model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=12,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )
    check_onnx(onnx_model_path)
    return

def torch_onnx_export_autoencoder_encode( model, onnx_model_path ):
    
    if not os.path.exists(onnx_model_path):
        
        #dynamic_axes = {
        #    "latent_model_input":   {0: "bs_x_2"},
        #    "prompt_embeds":        {0: "bs_x_2"},
        #    "noise_pred":           {0: "batch_size"}
        #}

        device = torch.device("cuda:0")
        dummy_inputs = {
            "input": torch.randn((1, 3, 256, 256),dtype=torch.float32).to(device),
        }
        output_names = ["moments"] # (device(type='cuda', index=0), torch.Size([1, 8, 32, 32]), torch.float16)

        #import apex
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                #with open(onnx_model_path, "wb") as f:
                with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
                    torch.onnx.export(
                        model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=12,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )
    check_onnx(onnx_model_path)
    return

def torch_onnx_export_midas( model, onnx_model_path ):
    if not os.path.exists(onnx_model_path):
        #dynamic_axes = {
        #    "latent_model_input":   {0: "bs_x_2"},
        #    "prompt_embeds":        {0: "bs_x_2"},
        #    "noise_pred":           {0: "batch_size"}
        #}

        device = torch.device("cuda:0")
        dummy_inputs = {
            "input": torch.randn((1, 3, 384, 384),dtype=torch.float32).to(device),
        }
        output_names = ["depth"] #torch.Size([1, 1, 384, 384]), torch.float16)

        #import apex
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                #with open(onnx_model_path, "wb") as f:
                with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
                    torch.onnx.export(
                        model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=12,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )
    check_onnx(onnx_model_path)
    return

def torch_onnx_export_pidinet( model, onnx_model_path ):
    if not os.path.exists(onnx_model_path):
        #dynamic_axes = {
        #    "latent_model_input":   {0: "bs_x_2"},
        #    "prompt_embeds":        {0: "bs_x_2"},
        #    "noise_pred":           {0: "batch_size"}
        #}

        device = torch.device("cuda:0")
        dummy_inputs = {
            "pidinet_input": torch.randn((1, 3, 384, 384),dtype=torch.float32).to(device),
        }
        output_names = ["sketch"] #torch.Size([1, 1, 384, 384]), torch.float16)

        #import apex
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                #with open(onnx_model_path, "wb") as f:
                with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
                    torch.onnx.export(
                        model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=12,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )
    check_onnx(onnx_model_path)
    return

def torch_onnx_export_cleaner( model, onnx_model_path ):
    if not os.path.exists(onnx_model_path):
        #dynamic_axes = {
        #    "latent_model_input":   {0: "bs_x_2"},
        #    "prompt_embeds":        {0: "bs_x_2"},
        #    "noise_pred":           {0: "batch_size"}
        #}

        device = torch.device("cuda:0")
        dummy_inputs = {
            "pidinet_input": torch.randn((1, 1, 384, 384),dtype=torch.float32).to(device),
        }
        output_names = ["sketch"] #torch.Size([1, 1, 384, 384]), torch.float16)

        #import apex
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                #with open(onnx_model_path, "wb") as f:
                with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
                    torch.onnx.export(
                        model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=12,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )
    check_onnx(onnx_model_path)
    return