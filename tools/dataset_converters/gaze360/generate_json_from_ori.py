import os
import json
from tqdm import tqdm
import shutil
import cv2
import time

ori_dataset_root = "/data/data4/zengwenzheng/data/gaze/gaze360/imgs/"
target_dataset_root = "/data/data4/zengwenzheng/data/gaze/gaze360/converted/"

splits = ['test', 'validation']
for split in splits:
    ori_gt_path = f'/data/data4/zengwenzheng/code/dataset_building/InstEye/InstEye/tools/dataset_converters/gaze360/{split}.txt'
    f = open(ori_gt_path)
    ori_gt_infos = f.readlines()
    vid_id = 0 # 第一个video_id是从1开始
    # pre_path_level_0 = ori_gt_info.split(' ')[0].split('/')[0]
    pre_path_level_0 = 0
    pre_path_level_2 = 0    
    pre_path_level_3 = 0
    new_frame_id = -1
    length = 0
    height = 0 
    width = 0

    dataset = {}
    info = {'info': {'description': 'converted_gaze360', 'url': '1', 'version': '1', 'year': '2022', 'contributor': 'Wenzheng Zeng', 'data_created': time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}   }
    licenses = {'licenses': 'only for research'}
    categories = {'categories': [{'supercategory': 'object', 'id': 1, 'name': 'person_face'}] }

    videos = []
    annotations = []
    anno_id = 1
    
    file_names = []

    # anno = {'length': 1, 'category_id': 1}
    anno_gaze = []

    ori_gt_infos.sort()
    for ori_gt_info in tqdm(ori_gt_infos):

        ori_gt_info = ori_gt_info.split(' ')
        ori_img_path = ori_gt_info[0].split('/')
        cur_path_level_0 = ori_img_path[0]
        cur_path_level_2 = ori_img_path[2]
        cur_path_level_3 = int(ori_img_path[3].strip('.jpg'))

        if (cur_path_level_3 != pre_path_level_3 + 1) or (cur_path_level_2 != pre_path_level_2) or (cur_path_level_0 != pre_path_level_0):
            # 如果第三级目录名称（帧序号）和之前的不连续，或者第0级或者第二级目录名称不同，则开启一个新的vid_id

            # 说明上一个video的信息收集已经结束了，先将其进行存储

            video = {'height': height, 'width': width, 'length': length, 'file_names': file_names, 'id': vid_id}
            videos.append(video)
            
            anno = {'height': height, 'width': width, 'length': 1, 'category_id': 1, 'gaze': anno_gaze, 'video_id': vid_id, 'id': vid_id} # 因为gaze360是单人，所以anno_id = vid
            annotations.append(anno)


            # 开启新的vid_id，清空length和file_names，以及anno

            vid_id += 1
            length = 0
            pre_path_level_3 = cur_path_level_3
            pre_path_level_2 = cur_path_level_2
            pre_path_level_0 = cur_path_level_0

            new_frame_id = -1   # 第一帧的index是0 
            # 新开一个json的子dict
            # 根据vid_id设定新的视频目录，把当前帧存入这个目录下，帧序号从0开始

            new_video_dir = os.path.join(target_dataset_root, f'{split}_rawframes', str(vid_id))
            os.makedirs(new_video_dir, exist_ok=True)
            file_names = []
            anno_gaze = []

            ########################################################################
            # 发现gaze360的每一帧裁剪的分辨率不同，而InstBlink代码接口默认一个视频内分辨率相同，为了统一，把gaze60每个视频的分辨率都统一成视频第一帧的分辨率，实际区别不大。
            first_img_path = os.path.join(ori_dataset_root, ori_gt_info[0])     #
            img = cv2.imread(first_img_path)
            height = img.shape[0]
            width = img.shape[1]
            #########################################################################


        new_frame_id += 1 
        length += 1
        new_relative_img_path = os.path.join(str(vid_id), str(new_frame_id).rjust(5, '0') + '.png')
        new_img_path = os.path.join(target_dataset_root, f'{split}_rawframes', new_relative_img_path)
        ori_img_path = os.path.join(ori_dataset_root, ori_gt_info[0])

        cur_img = cv2.imread(ori_img_path)

        if height != cur_img.shape[0] or width != cur_img.shape[1]:
        #     print('spatial resolution is not always equal') 
            cur_img = cv2.resize(cur_img, (width, height))

        # shutil.copyfile(ori_img_path, new_img_path)
        cv2.imwrite(new_img_path, cur_img)

        file_names.append(new_relative_img_path)

        cur_gaze = [float(ori_gt_info[1]), float(ori_gt_info[2]), float(ori_gt_info[3])]    
        anno_gaze.append(cur_gaze)

        pre_path_level_3 = cur_path_level_3
        pre_path_level_2 = cur_path_level_2
        pre_path_level_0 = cur_path_level_0



    # 把最后一个收集的video信息进行存储

    video = {'height': height, 'width': width, 'length': length, 'file_names': file_names, 'id': vid_id}
    videos.append(video)
    
    anno = {'height': height, 'width': width, 'length': 1, 'category_id': 1, 'gaze': anno_gaze, 'video_id': vid_id, 'id': vid_id} # 因为gaze360是单人，所以anno_id = vid
    annotations.append(anno)

    videos = videos[1:] # 由于代码的结构，第一个是空的，去掉
    annotations = annotations[1:]

    dataset.update(info)
    dataset.update(licenses)
    dataset.update({'videos': videos})
    dataset.update(categories)
    dataset.update({'annotations': annotations})

    final_json_root = os.path.join(target_dataset_root, 'annotations')
    os.makedirs(final_json_root, exist_ok=True)
    json.dump(dataset, open(os.path.join(final_json_root, f'{split}.json'), 'w'))
    print('Done')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))













