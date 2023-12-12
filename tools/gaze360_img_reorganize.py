import os
from tqdm import tqdm
import cv2
import time
import scipy.io as sio
import numpy as np
import torch

################### you should modify the path to raw gaze360 dataset  ############################
ori_dataset_root = "PATH to raw GAZE360 img dir eg. DataSet/gaze360/gaze360_dataset_htrht37t43t9723kdfnJKhf_v2/imgs"
meta_data_root = "PATH to raw GAZE360 metadata eg. DataSet/gaze360/gaze360_dataset_htrht37t43t9723kdfnJKhf_v2/metadata.mat"
origt = "PATH to raw GAZE360 dataset dir eg. DataSet/gaze360/gaze360_dataset_htrht37t43t9723kdfnJKhf_v2"
###################################################################################################
dataset_settings = ['FULL', 'L2CS']
target_dataset_roots = ['data/gaze360/', 'data/l2cs/']
splits = ['train','test']

for dataset_setting, target_dataset_root in zip(dataset_settings, target_dataset_roots):
    for split in splits:
        msg = sio.loadmat(meta_data_root)
        recordings = msg["recordings"]
        face_bbox = msg["person_face_bbox"]
        split_index = msg["split"]
        recording_index = msg["recording"]
        person_index = msg["person_identity"]
        frame_index = msg["frame"]
        total_num = recording_index.shape[1]
        ori_gt_path = os.path.join(origt, f"{split}.txt")
        f = open(ori_gt_path)
        ori_gt_infos = f.readlines()
        vid_id = 0
        pre_path_level_0 = 0
        pre_path_level_2 = 0
        pre_path_level_3 = 0
        new_frame_id = -1
        length = 0
        height = 0
        new_video_flag = 0
        width = 0
        un_face_det = 0
        total_frame = 0
        file_names = []
        ori_gt_infos.sort()

        point_index = 0

        for ori_gt_info in tqdm(ori_gt_infos):

            ori_gt_info = ori_gt_info.split(' ')
            ori_img_path = ori_gt_info[0].split('/')
            cur_path_level_0 = ori_img_path[0]
            cur_path_level_2 = ori_img_path[2]
            cur_path_level_3 = int(ori_img_path[3].strip('.jpg'))

            img_path_level_0 = recordings[0, recording_index[0, point_index]][0]
            img_path_level_2 = '%06d' % person_index[0, point_index]
            img_path_level_3 = int(frame_index[0, point_index])

            while (img_path_level_0 != cur_path_level_0) or (img_path_level_2 != cur_path_level_2) or (
                    img_path_level_3 != cur_path_level_3):
                # 没能成功匹配，index指针后移
                point_index = point_index + 1
                if point_index == 197588:
                    point_index = 0
                # 重新计算path
                img_path_level_0 = recordings[0, recording_index[0, point_index]][0]
                img_path_level_2 = '%06d' % person_index[0, point_index]
                img_path_level_3 = int(frame_index[0, point_index])

            if ((cur_path_level_3 != pre_path_level_3 + 1) or (cur_path_level_2 != pre_path_level_2) or (
                    cur_path_level_0 != pre_path_level_0) or new_video_flag == 1) and ((vid_id != 0 and length != 0 )or(vid_id == 0 )):
                # 如果第三级目录名称（帧序号）和之前的不连续，或者第0级或者第二级目录名称不同，则开启一个新的vid_id

                # 说明上一个video的信息收集已经结束了，先将其进行存储
                new_video_flag = 0
                # 开启新的vid_id，清空length和file_names，以及anno

                vid_id += 1
                # #TODO 弄个小数据集
                # if(vid_id==30):
                #     break
                length = 0
                pre_path_level_3 = cur_path_level_3
                pre_path_level_2 = cur_path_level_2
                pre_path_level_0 = cur_path_level_0

                new_frame_id = -1  # 第一帧的index是3 pad3帧
                # 新开一个json的子dict
                # 根据vid_id设定新的视频目录，把当前帧存入这个目录下，帧序号从0开始

                new_video_dir = os.path.join(target_dataset_root, f'{split}_rawframes', str(vid_id))
                os.makedirs(new_video_dir, exist_ok=True)
                file_names = []
                anno_gaze = []


                ########################################################################
                first_img_path = os.path.join(ori_dataset_root, ori_gt_info[0])  #
                img = cv2.imread(first_img_path)
                height = img.shape[0]
                width = img.shape[1]
                #########################################################################

            if dataset_setting =="L2CS":
                if (face_bbox[point_index] != np.array([-1, -1, -1, -1])).all():
                    new_frame_id += 1
                    length += 1
                    new_relative_img_path = os.path.join(str(vid_id), str(new_frame_id).rjust(5, '0') + '.png')
                    new_img_path = os.path.join(target_dataset_root, f'{split}_rawframes', new_relative_img_path)
                    ori_img_path = os.path.join(ori_dataset_root, ori_gt_info[0])

                    cur_img = cv2.imread(ori_img_path)  # 当前处理的图片

                    if height != cur_img.shape[0] or width != cur_img.shape[1]:
                        #     print('spatial resolution is not always equal')
                        cur_img = cv2.resize(cur_img, (width, height))  # resize成统一的分辨率

                    cv2.imwrite(new_img_path, cur_img)

                    file_names.append(new_relative_img_path.replace('\\', '/'))

                    cur_gaze = torch.tensor([float(ori_gt_info[1]), float(ori_gt_info[2]), float(ori_gt_info[3])])
                    total_frame += 1

                    pre_path_level_3 = cur_path_level_3
                    pre_path_level_2 = cur_path_level_2
                    pre_path_level_0 = cur_path_level_0

                else:
                    if length != 0:
                        new_video_flag = 1
                    else:
                        new_video_flag = 0
            else:
                new_frame_id += 1
                length += 1
                new_relative_img_path = os.path.join(str(vid_id), str(new_frame_id).rjust(5, '0') + '.png')
                new_img_path = os.path.join(target_dataset_root, f'{split}_rawframes', new_relative_img_path)
                ori_img_path = os.path.join(ori_dataset_root, ori_gt_info[0])

                cur_img = cv2.imread(ori_img_path)  # 当前处理的图片

                if height != cur_img.shape[0] or width != cur_img.shape[1]:
                    #     print('spatial resolution is not always equal')
                    cur_img = cv2.resize(cur_img, (width, height))  # resize成统一的分辨率

                cv2.imwrite(new_img_path, cur_img)

                file_names.append(new_relative_img_path.replace('\\', '/'))

                cur_gaze = torch.tensor([float(ori_gt_info[1]), float(ori_gt_info[2]), float(ori_gt_info[3])])
                total_frame += 1

                pre_path_level_3 = cur_path_level_3
                pre_path_level_2 = cur_path_level_2
                pre_path_level_0 = cur_path_level_0

        print('Done')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
