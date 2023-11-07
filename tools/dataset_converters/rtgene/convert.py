import json
import re
import numpy as np
from facenet_pytorch import MTCNN
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from tqdm import tqdm
import torch
print(torch.cuda.is_available())
test_fold1 = ["s001", "s002", "s008", "s010"]
test_fold2 = ["s003", "s004", "s007", "s009"]
test_fold3 = ["s005", "s006", "s011", "s012", "s013"]

train_fold1 = test_fold3+test_fold2
train_fold2 = test_fold1+test_fold3
train_fold3 = test_fold1+test_fold2

flods=[train_fold1,train_fold2,train_fold3,test_fold1,test_fold2,test_fold3]
name =["train1.json","train2.json","train3.json","test1.json","test2.json","test3.json"]

def GazeTo3d(gaze):
    x = -np.cos(gaze[1]) * np.sin(gaze[0])
    y = -np.sin(gaze[1])
    z = -np.cos(gaze[1]) * np.cos(gaze[0])
    return np.array([x, y, z])

name_id = 0
for flod in flods:
    dataset={}
    dataset_path = "/data/data3/zengwenzheng/gaze_work_202304/dataset/RT-GENE/"
    target_path = "/data/data3/zengwenzheng/gaze_work_202304/dataset/RT-GENE_convert/"
    detector = MTCNN(device='cuda')
    videos = []
    annotations = []
    anno_id = 1
    vid = 1
    categories = {'categories': [{'supercategory': 'object', 'id': 1, 'name': 'person_face'},
                                {'supercategory': 'object', 'id': 2, 'name': 'eye'},
                                {'supercategory': 'object', 'id': 3, 'name': 'head'}]}

    for f in flod:
        f = f+"_glasses"
        v_path = os.path.join(dataset_path, f)
        label_path = os.path.join(v_path, "label_combined.txt")
        v_path = os.path.join(v_path, "inpainted/face_after_inpainting")


        with open(label_path) as infile:
            label_info = infile.readlines()
        anno_gaze = []
        gaze_1 = {}
        for label in label_info:
            label = re.split('\[| |,|\]', label.strip())
            f_id = label[0]
            gaze = GazeTo3d(list(map(eval, [label[9], label[11]]))).tolist()
            gaze_1.update({f_id:gaze})
        #print(gaze_1)
        #print(len(os.listdir(v_path)))
        length = 0
        file_names = []
        head_bboxes = []
        face_bboxes = []
        eye_bboxes = []

        frame_id = 0
        gaze_it_id = 0
        if not os.path.exists(os.path.join(target_path, str(int(f[1:4]) ))):
            os.makedirs(os.path.join(target_path, str(int(f[1:4]))))
        frame_list = os.listdir(v_path)
        frame_list.sort()
        for frame in tqdm(frame_list):
            
            frame_path = os.path.join(v_path, frame)
            cur_img = cv2.imread(frame_path)
            cur_img = cv2.resize(cur_img,(112,112))
            if str(int(frame[:6])) not in gaze_1:
                continue
            face_bbox, score, landmark = detector.detect(cur_img, landmarks=True)
            if score[0]==None:
                continue
            face_bbox = face_bbox[0]
            landmark = landmark[0]
            if len(face_bbox) == 0:
                continue
            anno_gaze.append(gaze_1.get(str(int(frame[:6]))))
            length += 1
            head_bbox = [0, 0, 224, 224]
            face_bbox = [face_bbox[0],face_bbox[1],face_bbox[2]-face_bbox[0],face_bbox[3]-face_bbox[1]]
            face_bbox = [int(i) for i in face_bbox]
            eye_bbox = [landmark[0, 0] - 30,
                        min(landmark[0, 1], landmark[1, 1]) - 18,
                        landmark[1, 0] - landmark[0, 0] + 60, 36]
            eye_bbox = [int(i) for i in eye_bbox]
            head_bboxes.append(head_bbox)
            face_bboxes.append(face_bbox)
            eye_bboxes.append(eye_bbox)
            # cv2.rectangle(cur_img, (face_bbox[0:2]), (face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]),
            #               color=(255, 0, 0), thickness=2)
            # cv2.rectangle(cur_img, (eye_bbox[0:2]), (eye_bbox[0] + eye_bbox[2], eye_bbox[1] + eye_bbox[3]),
            #               color=(255, 0, 0), thickness=2)
            # cv2.imshow("00", cur_img)
            # cv2.waitKey()
            file_name = os.path.join(str(int(f[1:4])), str(frame_id).rjust(6, '0') + ".png")
            #print(os.path.join(target_path, file_name))
            cv2.imwrite(os.path.join(target_path, file_name), cur_img)
            # print(file_name)
            file_names.append(file_name)
            frame_id += 1

        video = {'height': 224, 'width': 224, 'length': length, 'file_names': file_names, 'id': vid}
        
        videos.append(video)
        anno1 = {'category_id': 1, 'gaze': anno_gaze,
                'bboxes': face_bboxes, 'video_id': vid,
                'id': anno_id}
        annotations.append(anno1)
        anno_id += 1

        anno2 = {'category_id': 2, 'gaze': anno_gaze,
                'bboxes': eye_bboxes, 'video_id': vid,
                'id': anno_id}
        annotations.append(anno2)
        anno_id += 1

        anno3 = {'category_id': 3, 'gaze': anno_gaze,
                'bboxes': head_bboxes, 'video_id': vid,
                'id': anno_id}
        annotations.append(anno3)
        anno_id += 1
        vid = vid+1

    dataset.update(categories)
    dataset.update({'videos':videos})
    dataset.update({'annotations': annotations})
    final_json_root = os.path.join(target_path, 'annotations')
    os.makedirs(final_json_root, exist_ok=True)
    json.dump(dataset, open(os.path.join(final_json_root, name[name_id]), 'w'))
    name_id+=1