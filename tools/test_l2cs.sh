#！/bin/bash
CUDA_VISIBLE_DEVICES="6" python tools/test_gaze360_gaze.py configs/multiclue_gaze/multiclue_gaze_r50_l2cs.py ckpts/multiclue_gaze_r50_l2cs.pth --json data/l2cs/test.json --root data/l2cs/test_rawframes/
python tools/calculate_mae_l2cs.py --evalfile results/results_multiclue_gaze_r50_l2cs_test.json