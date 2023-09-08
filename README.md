# videocomposer-TensorRT
Original source code copied from here, https://github.com/damo-vilab/videocomposer, thanks to damo-vilab for this great work. I perform GPU inference optimizing onto this baseline, via NVIDIA TensorRT, CUDA and other SDKs.

Currently, I just almost finished videocomposer inference pipeline optimization. the main optimizing part are AI model converting to TRT_fp16, converting gif video encoding to NVIDIA Codec SDK python version VPF(https://github.com/NVIDIA/VideoProcessingFramework).

I am still new to CUDA coding and inference optimization, any suggestions are all welcome.

Please follow here to run the baseline, https://github.com/damo-vilab/videocomposer/blob/main/README.md.