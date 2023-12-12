import torch
import math
from argparse import ArgumentParser
import json
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--evalfile', help='pred_gaze json file',
                        default="results/results_multiclue_gaze_r50_l2cs_test.json")
    parser.add_argument('--anno', help='annotation json file', default="data/l2cs/test.json")
    args = parser.parse_args()
    return args

def smooth_filter(input,alpha=0.6):
    # L2cs不能用smooth，因为有些不是连续的
    # alpha为1表示不进行smooth_filter(需要时设置为0.6)
    if input.size(0)>=2: # 输入至少2帧
        output = alpha * input
        output[0,:] += (1-alpha) * input[1,:]
        output[-1,:] += (1-alpha) * input[-2,:]
        output[1:-1,:] +=(1-alpha)*(input[0:-2,:]+input[2:,:])/2
        output = output / torch.norm(output,dim=1).unsqueeze(1)

    else:
        output = input

    return output

# 下面为以gaze360文章为标准的转换对（上面是原本的转换対

def yaw_pitch_to_vector(x):
    x = torch.reshape(x, (-1, 2))
    output = torch.zeros((x.size(0), 3))
    output[:,2] = - torch.cos(x[:,1]) * torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1]) * torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output


def vector_to_yaw_pitch(x):
    x = torch.reshape(x, (-1, 3))
    x = x / torch.norm(x, dim=1).reshape(-1, 1)
    output = torch.zeros((x.size(0), 2))
    output[:,0] = torch.atan2(x[:,0], - x[:,2])
    output[:,1] = torch.asin(x[:,1])
    return output


def compute_yaw_angular(target):
    if target.shape[-1] == 3:
        target = vector_to_yaw_pitch(target)
    target = target.view(-1, 2)
    output_yaw = 180 * torch.abs(target[:,0]) / math.pi
    return  output_yaw

def compute_pitch_angular(target):
    if target.shape[-1] == 3:
        target = vector_to_yaw_pitch(target)
    target = target.view(-1, 2)
    #temp = 180 * target[:,1] / math.pi
    output_pitch = 180 * torch.abs(target[:,1]) / math.pi
    return  output_pitch

def compute_angular_error(input, target):
    if input.shape[-1] == 2:
        input = yaw_pitch_to_vector(input)
    if target.shape[-1] == 2:
        target = yaw_pitch_to_vector(target)

    # input = smooth_filter(input)
    target =  target / torch.norm(target,dim=1).unsqueeze(1)
    input = input.view(-1, 3, 1)
    target = target.view(-1, 1, 3)
    output_dot = torch.bmm(target, input)  # 带batch的矩阵乘法，结果是（-1，1，1），也就是两个三维向量做内积
    output_dot = output_dot.view(-1)  # 变成一维tensor
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180 * torch.mean(output_dot) / math.pi  # 转换成角度
    # 计算两个三维向量的夹角 有 cos theta = 两向量点积除以两向量的模

    return  output_dot


def main(args):
    
    with open(args.evalfile, 'r') as eval_file:
        eval_data = json.load(eval_file)
    with open(args.anno, 'r') as anno_file:
        anno_data = json.load(anno_file)
    #gaze_error(eval_data, anno_data, 'gazes')
    gaze_error(eval_data, anno_data, 'fusion_gazes')
    # gaze_error(eval_data, anno_data, 'face_gazes')
    # gaze_error(eval_data, anno_data, 'eyes_gazes')
    # gaze_error(eval_data, anno_data, 'head_gazes')


def gaze_error(eval_data, anno_data, gaze_name):

    total_frame_360 = 0
    total_frame_front = 0
    total_frame_front_20 = 0
    total_angular_error_360 = 0
    total_angular_error_front = 0
    total_angular_error_front_20 = 0

    for anno_id, video in enumerate(eval_data):
        gaze_pred = video[gaze_name]
        gaze_gt = anno_data["annotations"][anno_id * 3]['gaze']
        video_len = len(gaze_pred)
        assert len(gaze_gt) == len(gaze_pred)

        gaze_front_gt = []
        gaze_front_pred = []

        gaze_front_20_gt = []
        gaze_front_20_pred = []

        front_len = 0
        front_20_len = 0
        gaze_pred = torch.tensor(gaze_pred)
        gaze_gt = torch.tensor(gaze_gt)
        
        gaze_pred = smooth_filter(gaze_pred)

        for i in range(gaze_gt.size(0)):
            # if anno_data["annotations"][anno_id]["detface"][i]:
            #     gaze_front_gt.append(gaze_gt[i])
            #     gaze_front_pred.append(gaze_pred[i])
            #     front_len += 1
            yaw_range = compute_yaw_angular(gaze_gt[i, :])
            pitch_range = compute_pitch_angular(gaze_gt[i, :])
            if yaw_range <= 90:
                gaze_front_gt.append(gaze_gt[i, :])
                gaze_front_pred.append(gaze_pred[i, :])
                front_len += 1  

            if yaw_range <= 20 and pitch_range <= 20:
                gaze_front_20_gt.append(gaze_gt[i, :])
                gaze_front_20_pred.append(gaze_pred[i, :])
                front_20_len += 1  

        angular_error_360 = compute_angular_error(gaze_pred, gaze_gt)
        # if angular_error_360  >50 :
        #     print("video_id:"+str(anno_id+1)+" error:"+str(angular_error_360.item())+" video_len "+str(video_len))
        # if video_len < 3:
        #     print(angular_error_360)
        total_frame_360 = total_frame_360 + video_len
        total_angular_error_360 = total_angular_error_360 + angular_error_360 * video_len

        if front_len > 0:
            gaze_front_gt = torch.stack(gaze_front_gt)
            gaze_front_pred = torch.stack(gaze_front_pred)
            angular_error_front = compute_angular_error(gaze_front_pred, gaze_front_gt)
            # if angular_error_front  > 30 :
            #  print("video_id:"+str(anno_id+1)+" error:"+str(angular_error_front.item())+" front_len "+str(front_len))
            total_frame_front = total_frame_front + front_len
            total_angular_error_front = total_angular_error_front + angular_error_front * front_len

        if front_20_len > 0:
            gaze_front_20_gt = torch.stack(gaze_front_20_gt)
            gaze_front_20_pred = torch.stack(gaze_front_20_pred)
            angular_error_front_20 = compute_angular_error(gaze_front_20_pred, gaze_front_20_gt)
            total_frame_front_20 = total_frame_front_20 + front_20_len
            total_angular_error_front_20 = total_angular_error_front_20 + angular_error_front_20 * front_20_len


    mean_angular_error_360 = total_angular_error_360 / total_frame_360
    mean_angular_error_360 = mean_angular_error_360.detach().item()

    mean_angular_error_front = total_angular_error_front / total_frame_front
    mean_angular_error_front = mean_angular_error_front.detach().item()

    mean_angular_error_front_20 = total_angular_error_front_20 / total_frame_front_20
    mean_angular_error_front_20 = mean_angular_error_front_20.detach().item()

    print("%s mean angular error 360: %.2f" % (gaze_name, mean_angular_error_360))
    print("%s mean angular front 90: %.2f" % (gaze_name, mean_angular_error_front))
    print("%s mean angular front 20: %.2f\n" % (gaze_name, mean_angular_error_front_20))


if __name__ == "__main__":
    args = parse_args()
    main(args)
