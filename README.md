# videocomposer-TensorRT
Original source code copied from here, https://github.com/damo-vilab/videocomposer, thanks to damo-vilab for this great work. I perform GPU inference optimizing onto this baseline, via NVIDIA TensorRT, CUDA and other SDKs.

Currently, I just almost finished videocomposer inference pipeline optimization. the main optimizing part are AI model converting to TRT_fp16, converting gif video encoding to NVIDIA Codec SDK python version VPF(https://github.com/NVIDIA/VideoProcessingFramework).

I am still new to CUDA coding and inference optimization, any suggestions are all welcome.

Please follow here to run the baseline, https://github.com/damo-vilab/videocomposer/blob/main/README.md.

# System config
This is my scripts just for reference:

> nvcc --version
> 
> apt-get update
> 
> #install python3.10
> 
> apt-get install software-properties-common
> 
> add-apt-repository ppa:deadsnakes/ppa
> 
> apt-get update
> 
> apt-get install python3.10
> 
> apt-get install vim
> 
> vim ~/.bashrc
> 
> #add "
> 
> alias python=python3.10
> 
> alias python3=python3.10
> 
> " to the end of file
> 
> source ~/.bashrc
> 
> apt-get install python3.10-distutils
> 
> python3.10 get-pip.py
> 
> pip install torch==1.12.0+cu113 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu113
> 
> pip install absl-py==1.4.0       aiohttp==3.8.4       aiosignal==1.3.1       aliyun-python-sdk-core==2.13.36       aliyun-python-sdk-kms==2.16.0       asttokens==2.2.1       async-timeout==4.0.2       attrs==22.2.0       backcall==0.2.0       cachetools==5.3.0       cffi==1.15.1       chardet==5.1.0       charset-normalizer==3.1.0       clean-fid==0.1.35       click==8.1.3       cmake==3.26.0       crcmod==1.7       cryptography==39.0.2       decorator==5.1.1       decord==0.6.0       easydict==1.10       einops==0.6.0       executing==1.2.0       fairscale==0.4.6       filelock==3.10.2       pytorch-lightning==1.4.2       flash-attn==0.2.0       frozenlist==1.3.3       fsspec==2023.3.0       ftfy==6.1.1       future==0.18.3       google-auth==2.16.2       google-auth-oauthlib==0.4.6       grpcio==1.51.3       huggingface-hub==0.13.3       idna==3.4       imageio==2.15.0       importlib-metadata==6.1.0       ipdb==0.13.13       ipython==8.11.0       jedi==0.18.2       jmespath==0.10.0       joblib==1.2.0       lazy-loader==0.2       markdown==3.4.3       markupsafe==2.1.2       matplotlib-inline==0.1.6       motion-vector-extractor==1.0.6       multidict==6.0.4       mypy-extensions==1.0.0       networkx==3.1       numpy==1.24.2       oauthlib==3.2.2       open-clip-torch==2.0.2       openai-clip==1.0.1       opencv-python==4.5.5.64       opencv-python-headless==4.7.0.68       oss2==2.17.0       packaging==23.0       parso==0.8.3       pexpect==4.8.0       pickleshare==0.7.5       pillow==9.4.0       pkgconfig==1.5.5       prompt-toolkit==3.0.38       protobuf==4.22.1       ptyprocess==0.7.0
> 
> pip install pure-eval==0.2.2       pyasn1==0.4.8       pyasn1-modules==0.2.8       pycparser==2.21       pycryptodome==3.17       pydeprecate==0.3.1       pygments==2.14.0       pynvml==11.5.0       pyre-extensions==0.0.23       pywavelets==1.4.1       pyyaml==6.0       regex==2023.3.22       requests==2.28.2       requests-oauthlib==1.3.1       rotary-embedding-torch==0.2.1       rsa==4.9       sacremoses==0.0.53       scikit-image==0.20.0       scikit-learn==1.2.2       scikit-video==1.1.11       scipy==1.9.1       simplejson==3.18.4       six==1.16.0       stack-data==0.6.2       tensorboard==2.12.0       tensorboard-data-server==0.7.0       tensorboard-plugin-wit==1.8.1       threadpoolctl==3.1.0       tifffile==2023.4.12       tokenizers==0.12.1       tomli==2.0.1
> 
> #install xformers
> 
> git clone https://github.com/facebookresearch/xformers/
> 
> cd xformers
> 
> git submodule update --init --recursive
> 
> pip install --verbose --no-deps -e .
> 
> pip install tqdm==4.65.0       traitlets==5.9.0       transformers==4.18.0       triton==2.0.0.dev20221120       typing-extensions==4.5.0       typing-inspect==0.8.0       urllib3==1.26.15       wcwidth==0.2.6       werkzeug==2.2.3      yarl==1.8.2       zipp==3.15.0
> 
> cd ..
> 
> cd videocomposer-TensorRT/
> 
> #cp all libGL* from host /usr/lib/x86_64-linux-gnu/
> 
> apt-get update
> 
> apt install ffmpeg
> 
> python run_net.py    --cfg configs/exp02_motion_transfer.yaml    --seed 9999    --input_video "demo_video/motion_transfer.mp4"    --image_path "demo_video/moon_on_water.jpg"    --input_text_desc "A beautiful big moon on the water at night"
> 
> #install nsys
> 
> cd /your/software/
> 
> ./NsightSystems-linux-public-2023.2.1.122-3259852.run
> 
> export PATH=/your/software/bin/:$PATH
> 
> #profiling with nsys
> 
> clear && /your/software/bin/nsys profile /usr/bin/python3.10 run_net.py --cfg configs/exp02_motion_transfer.yaml --seed 9999 --input_video "demo_video/motion_transfer.mp4" --image_path "demo_video/moon_on_water.jpg" --input_text_desc "A beautiful big moon on the water at night"
> 

# Install VPF

> cd ./videocomposer-TensorRT/VideoProcessingFramework
> 
> # or git clone from here, git clone https://github.com/NVIDIA/VideoProcessingFramework
> 
> cd ./VideoProcessingFramework
> 
> apt install -y libavfilter-dev libavformat-dev libavcodec-dev libswresample-dev libavutil-dev wget build-essential git
> 
> pip3 install .
> 
> apt update
> 
> apt install libtorch
> 
> pip install src/PytorchNvCodec
> 
> make run_samples_without_docker
> 
> find /usr -name _PytorchNvCodec.cpython-310-x86_64-linux-gnu.so
> 
> cp /usr/local/lib/python3.10/dist-packages/_PytorchNvCodec.cpython-310-x86_64-linux-gnu.so /usr/local/lib/python3.10/dist-packages/> PytorchNvCodec/
> 
> ldd /usr/local/lib/python3.10/dist-packages/PytorchNvCodec/_PytorchNvCodec.cpython-310-x86_64-linux-gnu.so
> 
> find /usr -name libtorch*
> 
> export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib/:$LD_LIBRARY_PATH
> 
> ldd /usr/local/lib/python3.10/dist-packages/PytorchNvCodec/_PytorchNvCodec.cpython-310-x86_64-linux-gnu.so
> 
> python ./samples/SamplePyTorch.py
> 
> clear && CUDA_VISIBLE_DEVICES=0,1 /usr/bin/python3.10 run_net.py --cfg configs/exp02_motion_transfer.yaml --seed 9999 --input_video > "demo_video/motion_transfer.mp4" --image_path "demo_video/moon_on_water.jpg" --input_text_desc "A beautiful big moon on the water at night"
> 
> cd ../videocomposer/VideoProcessingFramework/
> 
> clear && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.10 run_net.py --cfg configs/exp02_motion_transfer.yaml --seed 9999 --input_video "demo_video/> motion_transfer.mp4" --image_path "demo_video/moon_on_water.jpg" --input_text_desc "A beautiful big moon on the water at night"
> 
> cd ./VideoProcessingFramework/
> 
> clear && python ./tests/test_PyNvEncoder.py

# pre-load model weights
Reference https://github.com/damo-vilab/videocomposer to load pretrained models to ./videocomposer/model_weights, for me it looks like this:

> drwxrwxrwx 4 root root 4096 Jul 21 15:17 ./
> 
> drwxrwxrwx 13 root root 4096 Jul 21 15:28 ../
> 
> drwxrwxrwx 3 root root 4096 Jul 21 15:16 damo/
> 
> -rwxrwxrwx 1 root root 1372239913 Jul 21 15:16 midas_v3_dpt_large.pth*
> 
> -rwxrwxrwx 1 root root 5654395111 Jul 21 15:16 non_ema_228000.pth*
> 
> -rwxrwxrwx 1 root root 3944692325 Jul 21 15:17 open_clip_pytorch_model.bin*
> 
> -rwxrwxrwx 1 root root 288 Jul 21 15:16 readme.md*
> 
> -rwxrwxrwx 1 root root 177563291 Jul 21 15:17 sketch_simplification_gan.pth*
> 
> -rwxrwxrwx 1 root root 2871148 Jul 21 15:17 table5_pidinet.pth*
> 
> drwxrwxrwx 2 root root 4096 Jul 21 15:16 temp/
> 
> -rwxrwxrwx 1 root root 5214865159 Jul 21 15:17 v2-1_512-ema-pruned.ckpt*
>
# profiling Summary
please check the profiling/optimizing document for details, https://github.com/wujinzhong/videocomposer-TensorRT/blob/main/Videocomposer%20Pipeline%20Inference%20Performance%20Optimizing%20Project%20Overview.pdf.

In summary, we got 2.5X~3X speed up for AI models converting to TensorRT_fp16; VPF is of very high performance than CPU implementation video encoding, we use GPU hardware encoded H264 to replace original imageio CPU implementation.
![image](https://github.com/wujinzhong/videocomposer-TensorRT/assets/52945455/0d3ede7b-4416-473d-b43d-b9e32a074102)
![image](https://github.com/wujinzhong/videocomposer-TensorRT/assets/52945455/22af365e-0386-439c-af08-c4e8fcd8aa86)
