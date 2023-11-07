#！/bin/bash

CUDA_VISIBLE_DEVICES="1" python tools/train.py configs/multiclue_gaze/multiclue_gaze_r50_l2cs.py --gpu-id 0