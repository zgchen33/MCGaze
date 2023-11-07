import os
import cv2
# from pycocotools.my_bbox_ytvos import YTVOS
from mmdet.datasets.mpeblink_api import MPEblink
from mmdet.datasets.mpeblink_eval_api import MPEblinkEval
from tqdm import tqdm
import json
import time
import numpy as np




def visual_gt(dataset_root_0):
    dataset_root = os.path.join(dataset_root_0,'validate_rawframes')
    ann_file = os.path.join(dataset_root_0,'annotations/validate.json')
    visualization_save_path = os.path.join(dataset_root_0,'visual_test')
    mpeblink = MPEblink(ann_file)
    vid_ids = range(1, len(mpeblink.vids) + 1)
    font = cv2.FONT_HERSHEY_COMPLEX
    for vid_id in tqdm(vid_ids):
        if vid_id<34:
            continue
        ann_ids = mpeblink.getAnnIds(vidIds=[vid_id])
        ann_info = mpeblink.loadAnns(ann_ids)
        video_info = mpeblink.loadVids(vid_id)
        f = cv2.VideoWriter_fourcc(*'XVID')
        video_dir = os.path.join(visualization_save_path, str(vid_id))
        os.makedirs(video_dir, exist_ok=True)
        videoWriter = cv2.VideoWriter(os.path.join(visualization_save_path, f'demo_{vid_id}.avi'), f, 24,
                                      (video_info[0]['width'], video_info[0]['height']))
        frame_index = -1
        color = [(205,205,150), (185,218,255), (143,143,188),(225,225,225), (45,82,160),(125,125,125),(173,222,255),(240,255,240),(237,149,100),(92,92,205),(211,0,148), (144,238,144)]

        color_blink = (0, 255, 255)
        blink_count_list = [0] * len(ann_info)  # 用于存储每个instance的眨眼次数
        for img_path in video_info[0]['file_names']:    # 测试输出的可能不是完整视频，可能要改
            img = cv2.imread(os.path.join(dataset_root, img_path))
            person_index = 0
            frame_index+=1

            for person in ann_info:
                bbox = person['bboxes'].pop(0)  # 放到前面来，因为即使被跳过，这一帧的bboxes还是要吐出来
                if bbox == None:
                    person_index += 1
                    continue
                draw_color = color[person_index]
                for blink_event in person['blinks']:
                    if frame_index>=blink_event[0] and frame_index<=blink_event[1]:
                        draw_color = color_blink
                        if frame_index == blink_event[0]+2:
                            blink_count_list[person_index]+=1
                        break
                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),draw_color, 2)
                cv2.putText(img,f'gt{person_index} blink{blink_count_list[person_index]}',(int(bbox[0]),max(0,int(bbox[1])-10)),font,0.4,color[person_index],1)
                person_index+=1
            # os.makedirs(os.path.join(visualization_save_path, str(vid_id)),exist_ok=True)
            cv2.imwrite(os.path.join(visualization_save_path, img_path),img)
            videoWriter.write(img)
        videoWriter.release()


def form_video(sample_root, person_index):
    image_dir = f'{sample_root}/person{person_index}'
    target_save_path = f'{sample_root}/person{person_index}.avi'
    # os.system(f'ffmpeg -y -r 24 -f image2 -i {image_dir}/%05d.png -vcodec libx264 -pix_fmt yuv420p -acodec aac {target_save_path} -loglevel quiet')
    # print(1)
    img_paths = os.listdir(image_dir)
    img_paths.sort()
    img = cv2.imread(image_dir + '/' + img_paths[0])
    f = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(target_save_path, f, 24, (img.shape[1], img.shape[0]))
    for img_path in img_paths:
        cur_img = cv2.imread(image_dir + '/' + img_path)
        cur_img = cv2.resize(cur_img, (img.shape[1], img.shape[0]))
        # print(1)
        videoWriter.write(cur_img)

    videoWriter.release()

    image_dir = f'{sample_root}/person{person_index}blink'
    target_save_path = f'{sample_root}/person{person_index}blink.avi'

    img_paths = os.listdir(image_dir)
    img_paths.sort()
    img = cv2.imread(image_dir + '/' + img_paths[0])
    f = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(target_save_path, f, 24, (img.shape[1], img.shape[0]))
    for img_path in img_paths:
        cur_img = cv2.imread(image_dir + '/' + img_path)
        cur_img = cv2.resize(cur_img, (img.shape[1], img.shape[0]))
        # print(1)
        videoWriter.write(cur_img)

    videoWriter.release()

def visual_pred(data_root):
    dataset_root = os.path.join(data_root, 'train_rawframes')
    ann_file = os.path.join(data_root, 'annotations/train.json')
    # dataset_root = os.path.join(data_root, 'validate_rawframes_origin')
    # ann_file = os.path.join(data_root, 'annotations_origin/validate.json')
    # visualization_save_path = os.path.join(data_root, 'visual_test')
    visualization_save_path = os.path.join(data_root, 'visual_train_0.5_demo_filtered')
    # visualization_save_path = os.path.join(data_root, 'visual_test_by_origin_hungarian')
    mpeblink = MPEblink(ann_file)
    mpeblink_dets = mpeblink.loadRes("/data/data4/zengwenzheng/code/dataset_building/InstEye/InstEye/results/results_blink_converted.json")

    vid_ids = range(1,len(mpeblink_dets.vids)+1)
    font = cv2.FONT_HERSHEY_TRIPLEX

    # color = [(205,0,205), (225,225,225), (45,82,160),(125,125,125),(173,222,255),(240,255,240),(237,149,100),(92,92,205),(211,0,148), (144,238,144)]
    color = [(176, 196, 222), (255, 0, 255), (30, 144, 255), (250, 128, 114), (238, 232, 170),
             (255, 20, 147), (123, 104, 238), (255, 192, 203), (105, 105, 105), (85, 107, 47),
             (205, 133, 63), (0, 0, 128), (50, 205, 50), (127, 0, 127), (176, 48, 96),
             (128, 0, 0), (72, 61, 139), (0, 128, 0), (60, 179, 113), (0, 139, 139),
             (255, 0, 0), (255, 140, 0), (255, 215, 0), (0, 255, 0), (148, 0, 211),
             (0, 250, 154), (220, 20, 60), (0, 255, 255), (0, 191, 255), (0, 0, 255),
             (173, 255, 47), (218, 112, 214)]
    color_blink = (0, 255, 255)
    # choosed_vid = [8, 16]
    for vid_id in tqdm(vid_ids):
        # if vid_id<34:
        #     continue
        # print(1)
        # if vid_id not in choosed_vid:
        #     continue
        ann_ids = mpeblink_dets.getAnnIds(vidIds=[vid_id])
        ann_info = mpeblink_dets.loadAnns(ann_ids)
        # ann_info = sorted(ann_info, key=lambda x: x['score'], reverse=True)  # reverse的意思是按降序排列
        video_info = mpeblink_dets.loadVids(vid_id)
        f = cv2.VideoWriter_fourcc(*'XVID')
        video_dir = os.path.join(visualization_save_path, str(vid_id))
        os.makedirs(video_dir, exist_ok=True)
        videoWriter = cv2.VideoWriter(os.path.join(visualization_save_path,f'demo_{vid_id}.avi'), f, 24, (video_info[0]['width'],video_info[0]['height']))
        frame_index = -1

        blink_count_list = [0]*len(ann_info)    # 用于存储每个instance的眨眼次数
        for img_path in video_info[0]['file_names']:    # 测试输出的可能不是完整视频，可能要改
            img = cv2.imread(os.path.join(dataset_root, img_path))
            person_index = 0
            frame_index+=1

            for person in ann_info:
                bbox = person['bboxes'].pop(0)  # 放到前面来，因为即使被跳过，这一帧的bboxes还是要吐出来
                # score = person['score_per_img'].pop(0)  # 可能还是用mean好，这样实际上是一些重复检测
                # if person['score']<0.3: 暂时注释
                #     person_index += 1
                #     continue
                # if score<0.8: # 这块暂时注释掉了，显示linker的全部预测结果
                #      person_index += 1
                #      continue
                if bbox == None:
                    person_index += 1
                    continue
                draw_color = color[person_index]
                for blink_event in person['blinks_converted']:
                    if frame_index>=blink_event[0] and frame_index<=blink_event[1]:
                        draw_color = color_blink
                        # if frame_index == blink_event[0]:
                        # if frame_index == blink_event[0] + 2:
                        if frame_index == blink_event[1]:
                            blink_count_list[person_index]+=1
                        break

                cur_face_img = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2]), :]
                cur_empty_block = np.ones((250, 1200, 3)) * 255
                cur_empty_block = cur_empty_block.astype('uint8')
                cv2.putText(cur_empty_block, f'P{person_index} blink{blink_count_list[person_index]}', (100, 170), font, 6, (0, 0, 0), 10)
                #
                os.makedirs(os.path.join(visualization_save_path, str(vid_id) + 'info', 'person' + str(person_index)), exist_ok=True)
                os.makedirs(os.path.join(visualization_save_path, str(vid_id) + 'info', 'person' + str(person_index) + 'blink'), exist_ok=True)
                cv2.imwrite(os.path.join(visualization_save_path, str(vid_id) + 'info', 'person' + str(person_index) + 'blink', str(frame_index).rjust(5, '0') + '.png'), cur_empty_block)
                cv2.imwrite(os.path.join(visualization_save_path, str(vid_id) + 'info', 'person' + str(person_index), str(frame_index).rjust(5, '0') + '.png'), cur_face_img)


                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),draw_color, 5)
                cv2.putText(img,f'P{person_index} blink{blink_count_list[person_index]}',(int(bbox[0]),max(0,int(bbox[1])-10)),font,1.5,color[person_index],2)
                person_index+=1

            # os.makedirs(os.path.join(visualization_save_path, str(vid_id)),exist_ok=True)
            cv2.imwrite(os.path.join(visualization_save_path, img_path),img)
            videoWriter.write(img)
        videoWriter.release()
        sample_dir = os.path.join(visualization_save_path, str(vid_id) + 'info')
        for person_index in range(0, len(ann_info)):
            form_video(sample_dir, person_index)

if __name__ == '__main__':
    type = 'not_gt'
    # data_root = "/data/data1/zengwenzheng/code/dataset_building/BlinkTeViT/data/20220917_with_blink/"
    # data_root = "/data/data2/BlinkTeViT/data/20221029/"
    # data_root = "/data/data2/BlinkTeViT/data/20221029_resplit/"
    data_root = "/data/data4/zengwenzheng/data/gaze/gaze360/converted/"
    if type == 'gt':
        visual_gt(data_root)
    else:
        visual_pred(data_root)