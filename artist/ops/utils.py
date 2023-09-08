import os
import os.path as osp
import sys
import glob
import oss2 as oss
import math
import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
import binascii
import urllib.request
import zipfile
import gzip
import copy
import requests
import json
import hashlib
# import cv2
import numpy as np
import time
import base64
from io import BytesIO
from multiprocessing.pool import ThreadPool as Pool
from PIL import Image
import skvideo.io
from einops import rearrange
import imageio
import pickle
import logging
import sys
import os
from os.path import join, dirname
import PyNvCodec as nvc
from cuda import cuda, cudart
import videocomposer.trt_util_2
import nvtx


from videocomposer.artist import DOWNLOAD_TO_CACHE

__all__ = ['parse_oss_url',
           'parse_bucket',
           'read',
           'read_image',
           'read_gzip',
           'ceil_divide',
           'to_device',
           'put_object',
           'put_torch_object',
           'put_object_from_file',
           'get_object',
           'get_object_to_file',
           'rand_name',
           'save_image',
           'save_video',
           'save_video_vs_conditions',
           'save_video_multiple_conditions_with_data',
           'save_video_multiple_conditions_imageio',
           'save_video_multiple_conditions_VPF',
           'download_video_to_file',
           'save_video_grid_mp4',
           'save_caps',
           'ema',
           'parallel',
           'exists',
           'download',
           'unzip',
           'load_state_dict',
           'inverse_indices',
           'detect_duplicates',
           'read_tfs',
           'md5',
           'rope',
           'format_state',
           'breakup_grid',
           'huggingface_tokenizer',
           'huggingface_model']

TFS_CLIENT = None

def parse_oss_url(path):
    if path.startswith('oss://'):
        path = path[len('oss://'):]
    
    # configs
    configs = {
        'endpoint': os.getenv('OSS_ENDPOINT', None),
        'accessKeyID': os.getenv('OSS_ACCESS_KEY_ID', None),
        'accessKeySecret': os.getenv('OSS_ACCESS_KEY_SECRET', None),
        'securityToken': os.getenv('OSS_SECURITY_TOKEN', None)}
    bucket, path = path.split('/', maxsplit=1)
    if '?' in bucket:
        bucket, config = bucket.split('?', maxsplit=1)
        for pair in config.split('&'):
            k, v = pair.split('=', maxsplit=1)
            configs[k] = v
    
    # session
    session = parse_oss_url._sessions.setdefault(
        f'{bucket}@{os.getpid()}',
        oss.Session())
    
    # bucket
    bucket = oss.Bucket(
        auth=oss.Auth(configs['accessKeyID'], configs['accessKeySecret']),
        endpoint=configs['endpoint'],
        bucket_name=bucket,
        session=session)
    return bucket, path

parse_oss_url._sessions = {}

def parse_bucket(url):
    return parse_oss_url(osp.join(url, '_placeholder'))[0]

def read(filename, mode='r', retry=5):
    assert mode in ['r', 'rb']
    exception = None
    for _ in range(retry):
        try:
            if filename.startswith('oss://'):
                bucket, path = parse_oss_url(filename)
                content = bucket.get_object(path).read()
                if mode == 'r':
                    content = content.decode('utf-8')
            elif filename.startswith('http'):
                content = requests.get(filename).content
                if mode == 'r':
                    content = content.decode('utf-8')
            else:
                with open(filename, mode=mode) as f:
                    content = f.read()
            return content
        except Exception as e:
            exception = e
            continue
    else:
        raise exception

def read_image(filename, retry=5):
    exception = None
    for _ in range(retry):
        try:
            return Image.open(BytesIO(read(filename, mode='rb', retry=retry)))
        except Exception as e:
            exception = e
            continue
    else:
        raise exception


def download_video_to_file(filename, local_file, retry=5):
    exception = None
    for _ in range(retry):
        try:
            bucket, path = parse_oss_url(filename)
            bucket.get_object_to_file(path, local_file)
            break
        except Exception as e:
            exception = e
            continue
    else:
        raise exception
        
def read_gzip(filename, retry=5):
    exception = None
    for _ in range(retry):
        try:
            remove = False
            if filename.startswith('oss://'):
                bucket, path = parse_oss_url(filename)
                filename = rand_name(suffix=osp.splitext(filename)[1])
                bucket.get_object_to_file(path, filename)
                remove = True
            with gzip.open(filename) as f:
                content = f.read()
            if remove:
                os.remove(filename)
            return content
        except Exception as e:
            exception = e
            continue
    else:
        raise exception

def ceil_divide(a, b):
    return int(math.ceil(a / b))

def to_device(batch, device, non_blocking=False):
    if isinstance(batch, (list, tuple)):
        return type(batch)([
            to_device(u, device, non_blocking)
            for u in batch])
    elif isinstance(batch, dict):
        return type(batch)([
            (k, to_device(v, device, non_blocking))
            for k, v in batch.items()])
    elif isinstance(batch, torch.Tensor) and batch.device != device:
        batch = batch.to(device, non_blocking=non_blocking)
    return batch

def put_object(bucket, oss_key, data, retry=5):
    exception = None
    for _ in range(retry):
        try:
            return bucket.put_object(oss_key, data)
        except Exception as e:
            exception = e
            continue
    else:
        print(f'put_object to {oss_key} failed with error: {exception}', flush=True)

def put_torch_object(bucket, oss_key, data, retry=5):
    exception = None
    for _ in range(retry):
        try:
            buffer = BytesIO()
            torch.save(data, buffer)
            return bucket.put_object(oss_key, buffer.getvalue())
        except Exception as e:
            exception = e
            continue
    else:
        print(f'put_torch_object to {oss_key} failed with error: {exception}', flush=True)

def put_object_from_file(bucket, oss_key, filename, retry=5):
    exception = None
    for _ in range(retry):
        try:
            return bucket.put_object_from_file(oss_key, filename)
        except Exception as e:
            exception = e
            continue
    else:
        print(f'put_object_from_file to {oss_key} failed with error: {exception}', flush=True)

def get_object(bucket, oss_key, retry=5):
    exception = None
    for _ in range(retry):
        try:
            return bucket.get_object(oss_key).read()
        except Exception as e:
            exception = e
            continue
    else:
        print(f'get_object from {oss_key} failed with error: {exception}', flush=True)

def get_object_to_file(bucket, oss_key, filename, retry=5):
    exception = None
    for _ in range(retry):
        try:
            return bucket.get_object_to_file(oss_key, filename)
        except Exception as e:
            exception = e
            continue
    else:
        print(f'get_object_to_file from {oss_key} failed with error: {exception}', flush=True)

def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name

@torch.no_grad()
def save_image(bucket, oss_key, tensor, nrow=8, normalize=True, range=(-1, 1), retry=5):
    filename = rand_name(suffix='.jpg')
    for _ in [None] * retry:
        try:
            tvutils.save_image(tensor, filename, nrow=nrow, normalize=normalize, range=range)
            bucket.put_object_from_file(oss_key, filename)
            exception = None
            break
        except Exception as e:
            exception = e
            continue

    # remove temporary file
    if osp.exists(filename):
        os.remove(filename)
    if exception is not None:
        print('save image to {} failed, error: {}'.format(oss_key, exception), flush=True)

@torch.no_grad()
def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    tensor = tensor.permute(1,2,3,0)
    images = tensor.unbind(dim = 0)
    images = [(image.numpy()*255).astype('uint8') for image in images]
    imageio.mimwrite(path, images, fps=8)
    return images

#-----------------------------------------------------------------------------------

class cconverter:
    """
    Colorspace conversion chain.
    """

    #def __init__(self, width: int, height: int, gpu_id: int):
    def __init__(self, width: int, height: int, context: int, stream: int):
        #self.gpu_id = gpu_id
        self.w = width
        self.h = height
        self.chain = []
        self.context = context
        self.stream = stream

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> None:
        import PyNvCodec as nvc
        self.chain.append(
            #nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.gpu_id)
            nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.context, self.stream)
        )

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        import PyNvCodec as nvc
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        return surf.Clone() #surf.Clone(self.gpu_id)

def cuda_tensor_to_surface( tensor, gpu_id, torch_stream=None ):
    tensor_h = tensor.shape[0]
    tensor_w = tensor.shape[1]
    tensor_c = tensor.shape[2] #channel
    surf_dst = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, tensor_w, tensor_h, gpu_id)

    #print(f"tensor: {tensor.shape, tensor.device, tensor.dtype, tensor.stride(), tensor.storage_offset(), tensor.data_ptr()}")

    assert not surf_dst.Empty()
    dst_plane = surf_dst.PlanePtr()

    cuda.cuMemcpyDtoDAsync(dst_plane.GpuMem(), tensor.data_ptr(), tensor_h * tensor_w * tensor_c, torch_stream.cuda_stream)
    trt_util_2.synchronize(torch_stream)
    return surf_dst

    import pycuda.driver as pycuda

    pycuda.memcpy_dtod_async(dst_plane.GpuMem(), tensor.data_ptr(), 
                                    tensor_h * tensor_w * tensor_c)

    memcpy_2d = pycuda.Memcpy2D()
    memcpy_2d.width_in_bytes = tensor_w * tensor_c
    memcpy_2d.src_pitch = tensor_w * tensor_c
    memcpy_2d.dst_pitch = dst_plane.Pitch()
    memcpy_2d.width = tensor_h
    memcpy_2d.height = tensor_h
    print(f"memcpy pos0")
    memcpy_2d.set_src_device(tensor.data_ptr())
    print(f"memcpy pos1")
    memcpy_2d.set_dst_device(dst_plane.GpuMem())
    print(f"memcpy pos2")
    print(f"memcpy pos2, torch_stream.cuda_stream: {torch_stream.cuda_stream}")
    memcpy_2d(torch_stream.cuda_stream)
    print(f"memcpy pos3")

@torch.no_grad()
def video_tensor_to_imageio(tensor, path, duration = 120, loop = 0, optimize = True, torch_stream=None):
    tensor = tensor.permute(1,2,3,0)
    #print(f"tensor: {tensor.device, tensor.shape, tensor.dtype}")
    images = tensor.unbind(dim = 0)
    images = [(image.cpu().numpy()*255).astype('uint8') for image in images]
    imageio.mimwrite(path, images, fps=8)

    return images

@torch.no_grad()
def video_tensor_to_VPF(tensor, path, duration = 120, loop = 0, optimize = True, torch_stream=None,
                        testIdx=0):
    rng = nvtx.start_range(message="VPF0", color="red")
    tensor = tensor.permute(1,2,3,0)
    images = tensor.unbind(dim = 0)
    images_gpu = [(image*255).to(torch.uint8) for image in images]
    
    #for i, image in enumerate(images_gpu):
    #    print(f"image{i}: {image.dtype, image.shape}")

    if True: #VPF imp
        import PyNvCodec as nvc
        # Ground truth information about input video
        #gt_file = join(dirname(__file__), "test.mp4")
        #gt_file_res_change = join(dirname(__file__), "test_res_change.h264")
        gt_width = images_gpu[0].shape[1]
        gt_height = images_gpu[0].shape[0]
        gt_res_change = 47
        gt_is_vfr = False
        gt_pix_fmt = nvc.PixelFormat.NV12
        gt_framerate = 30
        gt_num_frames = 96
        gt_timebase = 8.1380e-5
        gt_color_space = nvc.ColorSpace.BT_709
        gt_color_range = nvc.ColorRange.MPEG
        gpu_id = 0

        #rawFrame = np.random.randint(0, 255, size=(gt_height, gt_width, 3), dtype=np.uint8)
        #nvDec = nvc.PyNvDecoder(gt_file, gpu_id)
        #c_nvUpl = nvc.PyFrameUploader(
        #        gt_width, gt_height, nvc.PixelFormat.RGB_PLANAR, c_ctx, c_str
        #    )

        rng_nv12 = nvtx.start_range(message="nv12", color='blue')
        
        # Retain primary CUDA device context and create separate stream per thread.
        _, c_ctx = cuda.cuCtxGetCurrent()
        c_str = torch_stream #torch.cuda.current_stream().cuda_stream
        
        to_nv12 = cconverter(gt_width, gt_height, c_ctx, c_str.cuda_stream)
        to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
        nvtx.end_range(rng_nv12)

        
        res = str(gt_width) + "x" + str(gt_height)
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)
        
        dstFile = open(path.split(".gif")[0]+f"_vpf_{testIdx}.mp4", "wb")

        pixel_format = nvc.PixelFormat.NV12
        profile = "high"
        surfaceformat = "nv12"
        if surfaceformat == 'yuv444':
            pixel_format = nvc.PixelFormat.YUV444
            profile = "high_444"
        elif surfaceformat == 'yuv444_10bit':
            pixel_format = nvc.PixelFormat.YUV444_10bit
            profile = "high_444_10bit"
        elif surfaceformat == 'yuv420_10bit':
            pixel_format = nvc.PixelFormat.YUV420_10bit
            profile = "high_420_10bit"
        
        rng_nvEnc = nvtx.start_range(message="nvc", color='blue')
        nvEnc = nvc.PyNvEncoder(
            {
                "preset": "P4",
                "tuning_info": "high_quality",
                "codec": "h264", #"hevc"
                "profile": profile, #"high"
                "s": res,
                "bitrate": "5M",
                'fps': '1', 
            },
            #gpu_id,
            c_ctx,
            c_str.cuda_stream,
            pixel_format, #nvc.PixelFormat.NV12
        )
        print(f"nvEnc: {nvEnc}")
        nvtx.end_range(rng_nvEnc)

        frames_sent = 0
        frames_recv = 0

        nvtx.end_range(rng)
        rng = nvtx.start_range(message="VPF1", color="red")
        while frames_sent<len(images_gpu):
            rawSurface = cuda_tensor_to_surface( images_gpu[frames_sent], gpu_id=0, torch_stream=torch_stream )
            
            dec_surf = rawSurface
            if not dec_surf or dec_surf.Empty():
                break
            dec_surf = to_nv12.run(dec_surf)
            
            frames_sent += 1

            nvEnc.EncodeSingleSurface(dec_surf, encFrame)
            if encFrame.size:
                frames_recv += 1

                byteArray = bytearray(encFrame)
                dstFile.write(byteArray)

        while True:
            success = nvEnc.FlushSinglePacket(encFrame)
            if success and encFrame.size:
                frames_recv += 1
            else:
                break

        assert frames_sent==frames_recv
        print(f"encode frames {frames_sent}")
        nvtx.end_range(rng)

    return None

@torch.no_grad()
def save_video(bucket, oss_key, tensor, mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],nrow=8, retry=5):
    mean=torch.tensor(mean,device=tensor.device).view(1,-1,1,1,1)#ncfhw
    std=torch.tensor(std,device=tensor.device).view(1,-1,1,1,1)#ncfhw
    tensor = tensor.mul_(std).add_(mean)  ####unnormalize back to [0,1]
    tensor.clamp_(0, 1)

    filename = rand_name(suffix='.gif')
    for _ in [None] * retry:
        try:
            one_gif = rearrange(tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            video_tensor_to_gif(one_gif, filename)
            bucket.put_object_from_file(oss_key, filename)
            exception = None
            break
        except Exception as e:
            exception = e
            continue

    # remove temporary file
    if osp.exists(filename):
        os.remove(filename)
    if exception is not None:
        print('save video to {} failed, error: {}'.format(oss_key, exception), flush=True)

@torch.no_grad()
def save_video_multiple_conditions_imageio(oss_key, video_tensor, model_kwargs, source_imgs, palette, 
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], nrow=8, retry=5, save_origin_video=True, bucket=None,
                    torch_stream=None):
    
    rng = nvtx.start_range( message="save_video_imageio", color="red" )
    mean=torch.tensor(mean,device=video_tensor.device).view(1, -1, 1, 1, 1)#ncfhw
    std=torch.tensor(std,device=video_tensor.device).view(1, -1, 1, 1, 1)#ncfhw
    video_tensor = video_tensor.mul_(std).add_(mean)  #### unnormalize back to [0,1]
    try:
        video_tensor.clamp_(0, 1)
    except:
        video_tensor = video_tensor.float().clamp_(0, 1)
    print(f"pos2, video_tensor: {video_tensor.device, video_tensor.shape, video_tensor.dtype}")
    video_tensor = video_tensor.cpu() 

    b, c, n, h, w = video_tensor.shape
    source_imgs = F.adaptive_avg_pool3d(source_imgs, (n, h, w))
    source_imgs = source_imgs.cpu()

    model_kwargs_channel3 = {}
    for key, conditions in model_kwargs[0].items():
        if conditions.shape[-1] == 1024:
            # Skip for style embeding
            continue
        if len(conditions.shape) == 3: # which means that it is histogram.
            conditions_np = conditions.cpu().numpy()
            conditions = []
            for i in conditions_np:
                vis_i = []
                for j in i:
                    vis_i.append(palette.get_palette_image(j, percentile=90, width=256, height=256))
                conditions.append(np.stack(vis_i))
            conditions = torch.from_numpy(np.stack(conditions)) # (8, 16, 256, 256, 3)
            conditions = rearrange(conditions, 'b n h w c -> b c n h w')
        else:
            if conditions.size(1) == 1:
                conditions = torch.cat([conditions, conditions, conditions], dim=1)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            if conditions.size(1) == 2:
                conditions = torch.cat([conditions, conditions[:,:1,]], dim=1)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.size(1) == 3:
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.size(1) == 4: # means it is a mask.
                color = ((conditions[:, 0:3] + 1.)/2.) # .astype(np.float32)
                alpha = conditions[:, 3:4] # .astype(np.float32)
                conditions = color * alpha + 1.0 * (1.0 - alpha)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        model_kwargs_channel3[key] = conditions.cpu() if conditions.is_cuda else conditions
    
    filename = oss_key # "output/output_" + rand_name(suffix='.gif')
    
    retry = 1
    for _ in [None] * retry:
        try:
            print(f"pos000")
            # set_trace()
            vid_gif = rearrange(video_tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            # con_gif = rearrange(conditions, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            cons_list = [rearrange(con, '(i j) c f h w -> c f (i h) (j w)', i = nrow) for _, con in model_kwargs_channel3.items()]
            source_imgs = rearrange(source_imgs, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            print(f"pos001")
            
            if save_origin_video:
                vid_gif = torch.cat([source_imgs,] + cons_list + [vid_gif,], dim=3)
            else:
                vid_gif = torch.cat(cons_list + [vid_gif,], dim=3)

            
            video_tensor_to_gif(vid_gif, filename)
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    if exception is not None:
        logging.info('save video to {} failed, error: {}'.format(oss_key, exception))
    nvtx.end_range(rng)

@torch.no_grad()
def save_video_multiple_conditions_VPF(oss_key, video_tensor, model_kwargs, source_imgs, palette, 
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], nrow=8, retry=5, save_origin_video=True, bucket=None,
                    torch_stream=None,
                    testIdx=0):
    
    rng = nvtx.start_range( message="presave", color="red" )
    #mean=torch.tensor(mean,device=video_tensor.device).view(1, -1, 1, 1, 1)#ncfhw
    #std=torch.tensor(std,device=video_tensor.device).view(1, -1, 1, 1, 1)#ncfhw
    mean = mean.to(video_tensor.device).view(1, -1, 1, 1, 1)#ncfhw
    std = std.to(video_tensor.device).view(1, -1, 1, 1, 1)#ncfhw

    video_tensor = video_tensor.mul_(std).add_(mean)  #### unnormalize back to [0,1]
    try:
        video_tensor.clamp_(0, 1)
    except:
        video_tensor = video_tensor.float().clamp_(0, 1)
    #video_tensor = video_tensor.cpu() 

    b, c, n, h, w = video_tensor.shape
    source_imgs = F.adaptive_avg_pool3d(source_imgs, (n, h, w))
    #source_imgs = source_imgs.cpu()
    
    nvtx.end_range(rng)
    model_kwargs_channel3 = {}
    rng0 = nvtx.start_range( message="conditions", color="red" )
    for key, conditions in model_kwargs[0].items():
        if conditions.shape[-1] == 1024:
            # Skip for style embeding
            continue
        if len(conditions.shape) == 3: # which means that it is histogram.
            conditions_np = conditions.cpu().numpy()
            conditions = []
            for i in conditions_np:
                vis_i = []
                for j in i:
                    vis_i.append(palette.get_palette_image(j, percentile=90, width=256, height=256))
                conditions.append(np.stack(vis_i))
            conditions = torch.from_numpy(np.stack(conditions)) # (8, 16, 256, 256, 3)
            conditions = rearrange(conditions, 'b n h w c -> b c n h w')
        else:
            if conditions.size(1) == 1:
                conditions = torch.cat([conditions, conditions, conditions], dim=1)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            if conditions.size(1) == 2:
                conditions = torch.cat([conditions, conditions[:,:1,]], dim=1)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.size(1) == 3:
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.size(1) == 4: # means it is a mask.
                color = ((conditions[:, 0:3] + 1.)/2.) # .astype(np.float32)
                alpha = conditions[:, 3:4] # .astype(np.float32)
                conditions = color * alpha + 1.0 * (1.0 - alpha)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        #model_kwargs_channel3[key] = conditions.cpu() if conditions.is_cuda else conditions
        model_kwargs_channel3[key] = conditions if conditions.is_cuda else conditions.to(torch.device("cuda:0"))

    nvtx.end_range(rng0)
    filename = oss_key # "output/output_" + rand_name(suffix='.gif')
    
    retry = 1
    for _ in [None] * retry:
        try:
            # set_trace()
            vid_gif = rearrange(video_tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            # con_gif = rearrange(conditions, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            cons_list = [rearrange(con, '(i j) c f h w -> c f (i h) (j w)', i = nrow) for _, con in model_kwargs_channel3.items()]
            source_imgs = rearrange(source_imgs, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            
            #vid_gif = vid_gif.to(device=torch.device('cuda'))
            cons_list = [con.to(device=torch.device('cuda')) for con in cons_list]
            #source_imgs = source_imgs.to(device=torch.device('cuda'))

            if save_origin_video:
                vid_gif = torch.cat([source_imgs,] + cons_list + [vid_gif,], dim=3)
            else:
                vid_gif = torch.cat(cons_list + [vid_gif,], dim=3)

            video_tensor_to_VPF(vid_gif, filename, torch_stream=torch_stream, testIdx=testIdx)
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    if exception is not None:
        logging.info('save video to {} failed, error: {}'.format(oss_key, exception))

@torch.no_grad()
def save_video_multiple_conditions_with_data(bucket, video_save_key, gt_video_save_key, vis_oss_key, video_tensor, model_kwargs, source_imgs, palette, 
                                   mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], nrow=8, retry=5):
    mean=torch.tensor(mean,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    std=torch.tensor(std,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    video_tensor = video_tensor.mul_(std).add_(mean)  #### unnormalize back to [0,1]
    video_tensor.clamp_(0, 1)

    b, c, n, h, w = video_tensor.shape
    source_imgs = F.adaptive_avg_pool3d(source_imgs, (n, h, w))
    source_imgs = source_imgs.cpu()

    model_kwargs_channel3 = {}
    for key, conditions in model_kwargs[0].items():
        # if key == 'masked':
        #     print(conditions.shape)
        #     print(conditions[:, 0:3].max(), conditions[:, 0:3].min())
        #     print(conditions[:, 3:].max(), conditions[:, 3:].min())
        if len(conditions.shape) == 3: # which means that it is histogram.
            conditions_np = conditions.cpu().numpy()
            conditions = []
            for i in conditions_np:
                vis_i = []
                for j in i:
                    vis_i.append(palette.get_palette_image(j, percentile=90, width=256, height=256))
                conditions.append(np.stack(vis_i))
            conditions = torch.from_numpy(np.stack(conditions)) # (8, 16, 256, 256, 3)
            conditions = rearrange(conditions, 'b n h w c -> b c n h w')
        else:
            if conditions.size(1) == 1:
                conditions = torch.cat([conditions, conditions, conditions], dim=1)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            if conditions.size(1) == 2:
                conditions = torch.cat([conditions, conditions[:,:1,]], dim=1)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.size(1) == 3:
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.size(1) == 4: # means it is a mask.
                color = ((conditions[:, 0:3] + 1.)/2.) # .astype(np.float32)
                alpha = conditions[:, 3:4] # .astype(np.float32)
                conditions = color * alpha + 1.0 * (1.0 - alpha)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        model_kwargs_channel3[key] = conditions.cpu() if conditions.is_cuda else conditions
    
    copy_video_tensor = video_tensor.clone()
    copy_source_imgs = source_imgs.clone()
    
    filename = rand_name(suffix='.gif')
    for _ in [None] * retry:
        try:
            vid_gif = rearrange(video_tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            # con_gif = rearrange(conditions, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            cons_list = [rearrange(con, '(i j) c f h w -> c f (i h) (j w)', j = nrow) for _, con in model_kwargs_channel3.items()]
            source_imgs = rearrange(source_imgs, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            vid_gif = torch.cat([source_imgs,] + cons_list + [vid_gif,], dim=3)
            
            video_tensor_to_gif(vid_gif, filename)
            bucket.put_object_from_file(vis_oss_key, filename)
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    
    # remove temporary file
    if osp.exists(filename):
        os.remove(filename)

    filename_pred = rand_name(suffix='.pkl')
    for _ in [None] * retry:
        try:
            copy_video_np = (copy_video_tensor.numpy() * 255).astype('uint8')
            pickle.dump(copy_video_np, open(filename_pred, 'wb'))
            bucket.put_object_from_file(video_save_key, filename_pred)
            break
        except Exception as e:
            print("error! ", video_save_key)
            continue
    
    # remove temporary file
    if osp.exists(filename_pred):
        os.remove(filename_pred)

    filename_gt = rand_name(suffix='.pkl')
    for _ in [None] * retry:
        try:
            copy_source_np = (copy_source_imgs.numpy() * 255).astype('uint8')
            pickle.dump(copy_source_np, open(filename_gt, 'wb'))
            bucket.put_object_from_file(gt_video_save_key, filename_gt)
            break
        except Exception as e:
            print("error! ", gt_video_save_key)
            continue
    
    # remove temporary file
    if osp.exists(filename_gt):
        os.remove(filename_gt)
    
    if exception is not None:
        print('save video to {} failed, error: {}'.format(vis_oss_key, exception), flush=True)


@torch.no_grad()
def save_video_vs_conditions(bucket, oss_key, video_tensor, conditions, source_imgs, mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],nrow=8, retry=5):
    mean=torch.tensor(mean,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    std=torch.tensor(std,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    video_tensor = video_tensor.mul_(std).add_(mean)  ####unnormalize back to [0,1]
    video_tensor.clamp_(0, 1)

    b, c, n, h, w = video_tensor.shape
    source_imgs = F.adaptive_avg_pool3d(source_imgs, (n, h, w))
    source_imgs = source_imgs.cpu()

    if conditions.size(1) == 1:
        conditions = torch.cat([conditions, conditions, conditions], dim=1)
        conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        # conditions = conditions * 255
    
    filename = rand_name(suffix='.gif')
    for _ in [None] * retry:
        try:
            vid_gif = rearrange(video_tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            con_gif = rearrange(conditions, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            source_imgs = rearrange(source_imgs, '(i j) c f h w -> c f (i h) (j w)', i = nrow)#num_sample_rows=8
            vid_gif = torch.cat([vid_gif, con_gif, source_imgs], dim=2)

            video_tensor_to_gif(vid_gif, filename)
            bucket.put_object_from_file(oss_key, filename)
            exception = None
            break
        except Exception as e:
            exception = e
            continue

    # remove temporary file
    if osp.exists(filename):
        os.remove(filename)
    if exception is not None:
        print('save video to {} failed, error: {}'.format(oss_key, exception), flush=True)



@torch.no_grad()
def save_video_grid_mp4(bucket, oss_key, tensor, mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5], nrow=None, fps=5, retry=5):
    mean=torch.tensor(mean,device=tensor.device).view(1,-1,1,1,1)#ncfhw
    std=torch.tensor(std,device=tensor.device).view(1,-1,1,1,1)#ncfhw
    tensor = tensor.mul_(std).add_(mean)  ####unnormalize back to [0,1]
    tensor.clamp_(0, 1)
    b, c, t, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 3, 4, 1)
    tensor = (tensor.cpu().numpy() * 255).astype('uint8')

    filename = rand_name(suffix='.mp4')
    for _ in [None] * retry:
        try:
            if nrow is None:
                nrow = math.ceil(math.sqrt(b))
            ncol = math.ceil(b / nrow)
            padding = 1
            video_grid = np.zeros((t, (padding + h) * nrow + padding,
                                (padding + w) * ncol + padding, c), dtype='uint8')
            for i in range(b):
                r = i // ncol
                c_ = i % ncol

                start_r = (padding + h) * r
                start_c = (padding + w) * c_
                video_grid[:, start_r:start_r + h, start_c:start_c + w] = tensor[i]
            skvideo.io.vwrite(filename, video_grid, inputdict={'-r': str(fps)})

            bucket.put_object_from_file(oss_key, filename)
            exception = None
            break
        except Exception as e:
            exception = e
            continue

    # remove temporary file
    if osp.exists(filename):
        os.remove(filename)
    if exception is not None:
        print('save video to {} failed, error: {}'.format(oss_key, exception), flush=True)

@torch.no_grad()
def save_text(bucket, oss_key, tensor, nrow=8, retry=5):
    len = tensor.shape[0]
    num_per_row = int(len/nrow)
    assert(len==nrow*num_per_row)
    texts = ''
    for i in range(nrow):
        for j in range(num_per_row):
            text = dec_bytes2obj(tensor[i*num_per_row+j])
            texts += text+'\n'
        texts += '\n'

    for _ in [None] * retry:
        try:
            bucket.put_object(oss_key, texts)
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    if exception is not None:
        print('save video to {} failed, error: {}'.format(oss_key, exception), flush=True)


@torch.no_grad()
def save_caps(bucket, oss_key, caps, retry=5):
    texts = ''
    for cap in caps:
        texts += cap
        texts += '\n'
    # print('save_caps ', texts)

    for _ in [None] * retry:
        try:
            bucket.put_object(oss_key, texts)
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    if exception is not None:
        print('save video to {} failed, error: {}'.format(oss_key, exception), flush=True)

@torch.no_grad()
def ema(net_ema, net, beta, copy_buffer=False):
    assert 0.0 <= beta <= 1.0
    for p_ema, p in zip(net_ema.parameters(), net.parameters()):
        p_ema.copy_(p.lerp(p_ema, beta))
    if copy_buffer:
        for b_ema, b in zip(net_ema.buffers(), net.buffers()):
            b_ema.copy_(b)

def parallel(func, args_list, num_workers=32, timeout=None):
    assert isinstance(args_list, list)
    if not isinstance(args_list[0], tuple):
        args_list = [(args, ) for args in args_list]
    if num_workers == 0:
        return [func(*args) for args in args_list]
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(func, args) for args in args_list]
        results = [res.get(timeout=timeout) for res in results]
    return results

def exists(filename):
    if filename.startswith('oss://'):
        bucket, path = parse_oss_url(filename)
        return bucket.object_exists(path)
    else:
        return osp.exists(filename)

def download(url, filename=None, replace=False, quiet=False):
    if filename is None:
        filename = osp.basename(url)
    if not osp.exists(filename) or replace:
        try:
            if url.startswith('oss://'):
                bucket, oss_key = parse_oss_url(url)
                bucket.get_object_to_file(oss_key, filename)
            else:
                urllib.request.urlretrieve(url, filename)
            if not quiet:
                print(f'Downloaded {url} to {filename}', flush=True)
        except Exception as e:
            raise ValueError(f'Downloading {filename} failed with error {e}')
    return osp.abspath(filename)

def unzip(filename, dst_dir=None):
    if dst_dir is None:
        dst_dir = osp.dirname(filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(dst_dir)

def load_state_dict(module, state_dict, drop_prefix=''):
    # find incompatible key-vals
    src, dst = state_dict, module.state_dict()
    if drop_prefix:
        src = type(src)([(k[len(drop_prefix):] if k.startswith(drop_prefix) else k, v) for k, v in src.items()])
    missing = [k for k in dst if not k in src]
    unexpected = [k for k in src if not k in dst]
    unmatched = [k for k in src.keys() & dst.keys() if src[k].shape != dst[k].shape]

    # keep only compatible key-vals
    incompatible = set(unexpected + unmatched)
    src = type(src)([(k, v) for k, v in src.items() if not k in incompatible])
    module.load_state_dict(src, strict=False)

    # report incompatible key-vals
    if len(missing) != 0:
        print('  Missing: ' + ', '.join(missing), flush=True)
    if len(unexpected) != 0:
        print('  Unexpected: ' + ', '.join(unexpected), flush=True)
    if len(unmatched) != 0:
        print('  Shape unmatched: ' + ', '.join(unmatched), flush=True)

def inverse_indices(indices):
    r"""Inverse map of indices.
        E.g., if A[indices] == B, then B[inv_indices] == A.
    """
    inv_indices = torch.empty_like(indices)
    inv_indices[indices] = torch.arange(len(indices)).to(indices)
    return inv_indices

def detect_duplicates(feats, thr=0.9):
    assert feats.ndim == 2

    # compute simmat
    feats = F.normalize(feats, p=2, dim=1)
    simmat = torch.mm(feats, feats.T)
    simmat.triu_(1)
    torch.cuda.synchronize()

    # detect duplicates
    mask = ~simmat.gt(thr).any(dim=0)
    return torch.where(mask)[0]

class TFSClient(object):

    def __init__(self, host='restful-store.vip.tbsite.net:3800', app_key='5354c9fae75f5'):
        self.host = host
        self.app_key = app_key

        # candidate servers
        self.servers = [u for u in read(f'http://{host}/url.list').strip().split('\n')[1:] if ':' in u]
        assert len(self.servers) >= 1
        self.__server_id = -1

    @property
    def server(self):
        self.__server_id = (self.__server_id + 1) % len(self.servers)
        return self.servers[self.__server_id]

    def read(self, tfs):
        tfs = osp.basename(tfs)
        meta = json.loads(read(f'http://{self.server}/v1/{self.app_key}/metadata/{tfs}?force=0'))
        img = Image.open(BytesIO(read(f'http://{self.server}/v1/{self.app_key}/{tfs}?offset=0&size={meta["SIZE"]}', 'rb')))
        return img

def read_tfs(tfs, retry=5):
    exception = None
    for _ in range(retry):
        try:
            global TFS_CLIENT
            if TFS_CLIENT is None:
                TFS_CLIENT = TFSClient()
            return TFS_CLIENT.read(tfs)
        except Exception as e:
            exception = e
            continue
    else:
        raise exception

def md5(filename):
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def rope(x):
    r"""Apply rotary position embedding on x of shape [B, *(spatial dimensions), C].
    """
    # reshape
    shape = x.shape
    x = x.view(x.size(0), -1, x.size(-1))
    l, c = x.shape[-2:]
    assert c % 2 == 0
    half = c // 2

    # apply rotary position embedding on x
    sinusoid = torch.outer(
        torch.arange(l).to(x),
        torch.pow(10000, -torch.arange(half).to(x).div(half)))
    sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    # reshape back
    return x.view(shape)

def format_state(state, filename=None):
    r"""For comparing/aligning state_dict.
    """
    content = '\n'.join([f'{k}\t{tuple(v.shape)}' for k, v in state.items()])
    if filename:
        with open(filename, 'w') as f:
            f.write(content)

def breakup_grid(img, grid_size):
    r"""The inverse operator of ``torchvision.utils.make_grid``.
    """
    # params
    nrow = img.height // grid_size
    ncol = img.width // grid_size
    wrow = wcol = 2  # NOTE: use default values here

    # collect grids
    grids = []
    for i in range(nrow):
        for j in range(ncol):
            x1 = j * grid_size + (j + 1) * wcol
            y1 = i * grid_size + (i + 1) * wrow
            grids.append(img.crop((x1, y1, x1 + grid_size, y1 + grid_size)))
    return grids

def huggingface_tokenizer(name='google/mt5-xxl', **kwargs):
    from transformers import AutoTokenizer  # v4.18.0 by default
    return AutoTokenizer.from_pretrained(DOWNLOAD_TO_CACHE(f'huggingface/tokenizers/{name}', name), **kwargs)

def huggingface_model(name='google/mt5-xxl', model_type='AutoModel', **kwargs):
    import transformers  # v4.18.0 by default
    return getattr(transformers, model_type).from_pretrained(DOWNLOAD_TO_CACHE(f'huggingface/models/{name}', name), **kwargs)
