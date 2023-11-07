#！/bin/bash

CUDA_VISIBLE_DEVICES="1" python tools/train.py configs/multiclue_gaze/multiclue_gaze_r50_gaze360.py --gpu-id 0