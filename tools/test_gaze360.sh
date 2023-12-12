#！/bin/bash

CUDA_VISIBLE_DEVICES="7" python tools/test_gaze360_gaze.py configs/multiclue_gaze/multiclue_gaze_r50_gaze360.py ckpts/multiclue_gaze_r50_gaze360.pth --json data/gaze360/test.json --root data/gaze360/test_rawframes/
python tools/calculate_mae_gaze360.py --evalfile results/results_multiclue_gaze_r50_gaze360_test.json
