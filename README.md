# videocomposer-TensorRT
Original source code copied from here, https://github.com/damo-vilab/videocomposer, thanks to damo-vilab for this great work. I perform GPU inference optimizing onto this baseline, via NVIDIA TensorRT, CUDA and other SDKs.

Currently, I just almost finished videocomposer inference pipeline optimization. the main optimizing part are AI model converting to TRT_fp16, converting gif video encoding to NVIDIA Codec SDK python version VPF(https://github.com/NVIDIA/VideoProcessingFramework).

I am still new to CUDA coding and inference optimization, any suggestions are all welcome.

Please follow here to run the baseline, https://github.com/damo-vilab/videocomposer/blob/main/README.md.

# Install VPF

> cd ./videocomposer-TensorRT/VideoProcessingFramework
> # or git clone from here, git clone https://github.com/NVIDIA/VideoProcessingFramework
> cd ./VideoProcessingFramework
> apt install -y libavfilter-dev libavformat-dev libavcodec-dev libswresample-dev libavutil-dev wget build-essential git
> pip3 install .
> apt update
> apt install libtorch
> pip install src/PytorchNvCodec
> make run_samples_without_docker
> find /usr -name _PytorchNvCodec.cpython-310-x86_64-linux-gnu.so
> cp /usr/local/lib/python3.10/dist-packages/_PytorchNvCodec.cpython-310-x86_64-linux-gnu.so /usr/local/lib/python3.10/dist-packages/> PytorchNvCodec/
> ldd /usr/local/lib/python3.10/dist-packages/PytorchNvCodec/_PytorchNvCodec.cpython-310-x86_64-linux-gnu.so
> find /usr -name libtorch*
> export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib/:$LD_LIBRARY_PATH
> ldd /usr/local/lib/python3.10/dist-packages/PytorchNvCodec/_PytorchNvCodec.cpython-310-x86_64-linux-gnu.so
> python ./samples/SamplePyTorch.py
> clear && CUDA_VISIBLE_DEVICES=0,1 /usr/bin/python3.10 run_net.py --cfg configs/exp02_motion_transfer.yaml --seed 9999 --input_video > "demo_video/motion_transfer.mp4" --image_path "demo_video/moon_on_water.jpg" --input_text_desc "A beautiful big moon on the water at night"
> cd ../videocomposer/VideoProcessingFramework/
> clear && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.10 run_net.py --cfg configs/exp02_motion_transfer.yaml --seed 9999 --input_video "demo_video/> motion_transfer.mp4" --image_path "demo_video/moon_on_water.jpg" --input_text_desc "A beautiful big moon on the water at night"
> cd ./VideoProcessingFramework/
> clear && python ./tests/test_PyNvEncoder.py