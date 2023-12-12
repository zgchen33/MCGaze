import json
import os
import os.path as osp
from argparse import ArgumentParser
from threading import Thread
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import collate, scatter
from tqdm import tqdm
import math
from mmdet.apis import init_detector
from mmdet.core.bbox import bbox_overlaps
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods, flops_to_string, params_to_string
import time
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint',help='Checkpoint file')
    parser.add_argument(
        '--json',
        default="data/gaze360/test.json",help='Path to gaze test json file')   
    parser.add_argument(
        '--root', default="data/gaze360/test_rawframes/", help='Path to image file')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))

def main(args):
    model = init_detector(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options) # 这个函数内部调用了build_detector
    model = add_flops_counting_methods(model)
    cfg = model.cfg
    anno = json.load(open(args.json))
    test_pipeline = Compose(cfg.data.test.pipeline)

    results = []
    clip_len = 7   # 定义单次前传的clip_len
    stride = 4  # 定义stride
    videoes_forward_time = 0
    all_video_length = 0
    person_threshold = 0.5
    for video in tqdm(anno['videos']):
        imgs = video['file_names']
        video_det_bboxes = []
        video_det_other_gaze = []
        datas, threads = [], []
        video_length = len(imgs)

        if video_length <= clip_len:
            clip_num = 1
        else:
            clip_num = math.ceil((video_length-clip_len)/stride) + 1
        for clip_index in range(0, clip_num):
            if clip_index!=clip_num-1:  # 判断是否为最后一个切片，如果不是，则正常按stride切片出来
                cur_clip = imgs[clip_index*stride:clip_index*stride + clip_len]
                clip_overlap = clip_len - stride
            else:   # 如果是最后一个切片，则倒着取最后clip_num帧
                cur_clip = imgs[-clip_len:]
                if (video_length-clip_len)%stride:
                    clip_overlap = clip_len - (video_length-clip_len)%stride
                else:
                    clip_overlap = clip_len - stride
            threads = []
            datas = []
            for img in cur_clip:
                data = dict(img_info=dict(filename=img), img_prefix=args.root)
                threads.append(Thread(target=load_datas, args=(data, test_pipeline, datas))) # Thread是多线程，target是实际被调用的函数，args是调用target函数需要的参数
                threads[-1].start()
            for thread in threads:
                thread.join()

            datas = sorted(datas, key=lambda x:x['img_metas'].data['filename']) # 按帧顺序 img名称从小到大

            datas = collate(datas, samples_per_gpu=len(cur_clip)) # 用来形成batch用的
            datas['img_metas'] = datas['img_metas'].data
            datas['img'] = datas['img'].data
            datas = scatter(datas, [args.device])[0]

            with torch.no_grad():
                model.start_flops_count()

                start_time = time.time()
                (det_bboxes, det_labels), det_gazes = model(
                    return_loss=False,
                    rescale=True,
                    format=False,# 返回的bbox既包含face_bboxes也包含head_bboxes
                    **datas)    # 返回的bbox格式是[x1,y1,x2,y2],根据return_loss函数来判断是forward_train还是forward_test.
                end_time = time.time()
                total_forward = (end_time - start_time) #/ clip_len
                # if clip_len == 7:
                #     videoes_forward_time += total_forward
                #     all_video_length += clip_len
                # print('Total forward time is %4.4f seconds' % total_forward)

                det_bboxes = torch.stack(det_bboxes) # (7, 3, 5)
                gaze_dim = det_gazes['gaze_score'].size(1)
                det_fusion_gaze = det_gazes['gaze_score'].view((det_gazes['gaze_score'].shape[0], 1, gaze_dim))
                det_other_gaze = torch.cat((det_gazes['face_gaze_score'].view(det_gazes['face_gaze_score'].shape[0], 1, gaze_dim),
                                             det_gazes['eyes_gaze_score'].view(det_gazes['eyes_gaze_score'].shape[0], 1, gaze_dim),
                                             det_gazes['head_gaze_score'].view(det_gazes['head_gaze_score'].shape[0], 1, gaze_dim)),
                                             dim=1)     # (7, 3, 3)            
                model.stop_flops_count()

            # 进行clip间的匹配
            if clip_index!=0:  # 如果不是第一个clip
                previous_det_bboxes_for_match = video_det_bboxes[:,-clip_overlap:,:]

                det_bboxes = det_bboxes.permute(1, 0, 2)# 执行完此行后(3,clip,5), 原本size为(clip,不同类型框数(3),5) 
                det_other_gaze = det_other_gaze.permute(1, 0, 2)# 执行完此行后(3,clip,3), 原本size为(clip,3,3)
                det_fusion_gaze = det_fusion_gaze.permute(1, 0, 2)
                # 下面几行为处理bbox_scores < person_threshold，赋值zeros的过程
                bbox_coords, bbox_scores = torch.split(det_bboxes, [4, 1], dim=-1)
                # 使用torch.where()函数构造掩码张量，然后用torch.zeros()函数来填充那些置信度小于person_threshold的位置
                mask = bbox_scores < person_threshold
                bbox_coords = torch.where(mask, torch.zeros_like(bbox_coords), bbox_coords)
                # 将bbox_coords和bbox_scores合并成一个张量
                det_bboxes = torch.cat([bbox_coords, bbox_scores], dim=-1)

                previous_person_num = previous_det_bboxes_for_match.size(0)
                # cur_person_num = det_bboxes.size(0)
                # mat = np.zeros([previous_person_num, cur_person_num])

                next_padding_bboxes = torch.zeros([previous_person_num,clip_len-clip_overlap,5]).to(video_det_bboxes.device) # padding,帧数为clip_len-clip_overlap,也就是即将扩展的帧长度,理论上，bbox=0算出来的iou应该都是0吧？
                video_det_bboxes = torch.cat((video_det_bboxes, next_padding_bboxes),1)
                # next_padding_labels = torch.zeros()
                next_padding_gazes = torch.zeros([previous_person_num,clip_len-clip_overlap,gaze_dim]).to(video_det_other_gaze.device)
                video_det_other_gaze = torch.cat((video_det_other_gaze, next_padding_gazes),1)
                # fusion_gaze's padding
                next_padding_fusion_gazes = torch.zeros([det_fusion_gaze.shape[0],clip_len-clip_overlap,gaze_dim]).to(video_det_fusion_gaze.device)
                video_det_fusion_gaze = torch.cat((video_det_fusion_gaze, next_padding_fusion_gazes),1)
                
#################################################因为数据集只有一个人，不需要匹配是否是同个人！！！###################################################
                # 未重叠的部分直接赋值给video_det；重叠部分下面会处理
                video_det_bboxes[:, -(clip_len-clip_overlap):, :] = det_bboxes[:, -(clip_len-clip_overlap):, :] # 这里有可能会有问题
                video_det_other_gaze[:, -(clip_len-clip_overlap):, :] = det_other_gaze[:, -(clip_len-clip_overlap):, :]
                video_det_fusion_gaze[:, -(clip_len-clip_overlap):, :] = det_fusion_gaze[:, -(clip_len-clip_overlap):, :]

                # 因为可能存在重叠部分在上次或这次检测时为none_bbox，所以这里bbox取平均是不合理的，要写成逐祯处理
                overlap_bboxs1 = video_det_bboxes[:, -clip_len:-(clip_len-clip_overlap), :]
                overlap_bboxs2 = det_bboxes[:, -clip_len:-(clip_len-clip_overlap), :]

                bbox_coords1, bbox_scores1 = torch.split(overlap_bboxs1, [4, 1], dim=-1)
                bbox_coords2, bbox_scores2 = torch.split(overlap_bboxs2, [4, 1], dim=-1)

                # 使用torch.where()函数构造掩码张量，然后用torch.zeros()函数来填充那些置信度小于person_threshold的位置
                mask1 = bbox_scores1 < person_threshold
                mask2 = bbox_scores2 < person_threshold

                mask = torch.logical_or(mask1, mask2)
                # 此处先求平均，下面会利用mask对 上次或这次 检测不满足 score > person_threshold (即为none_bbox的赋为zeros)
                bbox_coords = (bbox_coords1 + bbox_coords2) / 2
                bbox_scores = (bbox_scores1 + bbox_scores2) / 2
                bbox_coords = torch.where(mask, torch.zeros_like(bbox_coords), bbox_coords)
                # 将bbox_coords和bbox_scores合并成一个张量
                overlap_bboxes = torch.cat([bbox_coords, bbox_scores], dim=-1)
                # 重叠部分处理完，赋值给video_det_bboxes
                video_det_bboxes[:, -clip_len:-(clip_len-clip_overlap), :] = overlap_bboxes
                video_det_other_gaze[:, -clip_len:-(clip_len-clip_overlap), :] = (video_det_other_gaze[:, -clip_len:-(clip_len-clip_overlap), :] + det_other_gaze[:, -clip_len:-(clip_len-clip_overlap), :])/2
                video_det_fusion_gaze[:, -clip_len:-(clip_len-clip_overlap), :] = (video_det_fusion_gaze[:, -clip_len:-(clip_len-clip_overlap), :] + det_fusion_gaze[:, -clip_len:-(clip_len-clip_overlap), :])/2
#################################################因为数据集只有一个人，不需要匹配是否是同个人！！！###################################################      
            
            else: # 第一个video_cilp

                det_bboxes = det_bboxes.permute(1, 0, 2)# 原本size为(clip,不同类型框数(3),5) 执行完此行后(3,clip,5)
                det_other_gaze = det_other_gaze.permute(1, 0, 2)# 原本size为(clip,3,3) 执行完此行后(3,clip,3)
                det_fusion_gaze = det_fusion_gaze.permute(1, 0, 2)

                bbox_coords, bbox_scores = torch.split(det_bboxes, [4, 1], dim=-1)
                # 使用torch.where()函数构造掩码张量，然后用torch.zeros()函数来填充那些置信度小于person_threshold的位置
                mask = bbox_scores < person_threshold
                bbox_coords = torch.where(mask, torch.zeros_like(bbox_coords), bbox_coords)
                # bbox_scores = torch.where(mask, torch.zeros_like(bbox_scores), bbox_scores)
                # 将bbox_coords和bbox_scores合并成一个张量
                video_det_bboxes = torch.cat([bbox_coords, bbox_scores], dim=-1)

                video_det_other_gaze = det_other_gaze
                video_det_fusion_gaze = det_fusion_gaze


        det_bboxes =  video_det_bboxes.permute(1,0,2)# 执行完此行后(len(img),不同类型框数3,5)，原本size为(不同类型框数3,len(img),5)
        det_other_gaze = video_det_other_gaze.permute(1,0,2)
        det_fusion_gaze = video_det_fusion_gaze.permute(1,0,2)


        #for inst_ind in range(det_bboxes.size(1)):  # 遍历获取的top10的query信息
        objs = dict(
            video_id=video['id'],
            category_id=1,
            fusion_gazes=[],

            face_bboxes=[],
            face_gazes=[],
            face_score=[],

            eyes_bboxes=[],
            eyes_gazes=[],
            eyes_score=[],

            head_bboxes=[],
            head_gazes=[],
            head_score=[],
            )  # 获取当前遍历第inst_ind个query的score信息（帧间取平均）其实这块有点不合理，不应该按mean来取，应该用.max取，因为有些instance并不是每帧都出现
        for sub_ind in range(det_bboxes.size(0)):   # 遍历每一帧的
            # fusion_gaze
            objs['fusion_gazes'].append(det_fusion_gaze[sub_ind,0,:].cpu().numpy().tolist()) # 这个最终的数据类型有待商榷e

            # face部分
            face_m = det_bboxes[sub_ind, 0, :-1].detach().cpu().numpy().tolist()
            if (face_m[0] + face_m[1] + face_m[2] + face_m[3]) == 0:
                face_m = None
            else:
                face_m = [face_m[0],face_m[1],face_m[2]-face_m[0],face_m[3]-face_m[1]]
            objs['face_bboxes'].append(face_m)
            objs['face_gazes'].append(det_other_gaze[sub_ind,0,:].cpu().numpy().tolist()) # 这个最终的数据类型有待商榷
            objs['face_score'].append(det_bboxes[sub_ind,0,-1].item())

            # eyes部分
            eyes_m = det_bboxes[sub_ind, 1, :-1].detach().cpu().numpy().tolist()
            if (eyes_m[0] + eyes_m[1] + eyes_m[2] + eyes_m[3]) == 0:
                eyes_m = None
            else:
                eyes_m = [eyes_m[0],eyes_m[1],eyes_m[2]-eyes_m[0],eyes_m[3]-eyes_m[1]]
            objs['eyes_bboxes'].append(eyes_m)
            objs['eyes_gazes'].append(det_other_gaze[sub_ind,1,:].cpu().numpy().tolist()) # 这个最终的数据类型有待商榷
            objs['eyes_score'].append(det_bboxes[sub_ind,1,-1].item())

            # head部分
            head_m = det_bboxes[sub_ind, 2, :-1].detach().cpu().numpy().tolist()
            if (head_m[0] + head_m[1] + head_m[2] + head_m[3]) == 0:
                head_m = None
            else:
                head_m = [head_m[0],head_m[1],head_m[2]-head_m[0],head_m[3]-head_m[1]]
            objs['head_bboxes'].append(head_m)
            objs['head_gazes'].append(det_other_gaze[sub_ind,2,:].cpu().numpy().tolist()) # 这个最终的数据类型有待商榷
            objs['head_score'].append(det_bboxes[sub_ind,2,-1].item())
        results.append(objs)
    
    # average_video_forward_time = videoes_forward_time / all_video_length
    # print('video_forward_time is {} seconds, num_frame is {}'.format(videoes_forward_time, all_video_length))
    # print('average_video_forward_time is %4.4f seconds' % average_video_forward_time)

    # export results to json format and calculate mean Average-Precision
    os.makedirs('results',exist_ok=True)
    write_path = os.path.join('results', f'results_{args.config.rstrip(".py").split("/")[-1]}_{args.json.split("/")[-1]}')
    json.dump(results, open(write_path, 'w'))
    print('Done')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

if __name__ == '__main__':
    args = parse_args()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    main(args)
