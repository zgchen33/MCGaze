# 使用 头-脸-眼 多重线索的时空交互进行端到端的视频视线估计（多重线索视线估计）
<div align="center">

管一然\*，陈卓光 \*，[曾文正](https://wenzhengzeng.github.io/)<sup>†</sup>，[曹治国](https://scholar.google.com/citations?user=396o2BAAAAAJ)， [肖阳](https://scholar.google.com/citations?user=NeKBuXEAAAAJ)<sup>†</sup> 

华中科技大学 

*：同等贡献，†：通讯作者

</div>


<div align="center">
  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/end-to-end-video-gaze-estimation-via/gaze-estimation-on-gaze360)](https://paperswithcode.com/sota/gaze-estimation-on-gaze360?p=end-to-end-video-gaze-estimation-via)

<img src="demo/d3_n.gif" width="50%"/><img src="demo/d2_n.gif" width="50%"/>

[English](README.md)|简体中文

</div>

## 介绍

本代码库包含了论文“使用 头-脸-眼 多重线索的时空交互进行端到端的视频视线估计”的官方实现。

<div align="center">

<img src="./fig1.png">

</div>

我们提出通过以端到端学习的方式捕获头-脸-眼之间的时空交互关系来提升视频视线估计的效果估计。在具有挑战性的 Gaze360 数据集上进行的实验验证了我们的方法的优越性。
## 实验结果及模型

在我们的工作中，我们在两种不同的数据集设置（Gaze360-setting 和 [l2CS-setting](https://github.com/Ahmednull/L2CS-Net)（只考虑可检测到人脸的样本））中测试我们的模型，以便与以前的方法进行公平比较。

您可以从表内的链接下载模型的checkpoint。
| 数据集设置                     | 骨干网络 | 平均角度误差-正面180                                   | 权重 |
| :------------------------: | :-------: | :------------------------------------: | :----------------------: |
| Gaze360-setting   | R-50    |  10.74            |           [谷歌网盘](https://drive.google.com/file/d/1ru0xhuB5N9kwvN9XLvZMQvVSfOgtbxmq/view?usp=share_link)        |
| l2cs-setting      | R-50    | 9.81        |         [谷歌网盘](https://drive.google.com/file/d/1frp_rmER8_hf2xC0hbtjRTLA4TBqYePq/view?usp=share_link)    |        
  
## 使用本代码库
### 准备你的python虚拟环境
1. 创建一个新的conda环境

   ```bash
   conda create -n MCGaze python=3.9
   conda activate MCGaze
   ```
   
2. 安装 Pytorch (推荐使用1.7.1 ), scipy, tqdm, pandas。
   ```bash
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   pip install scip tqdm pandas
   ```

4. 安装 MMDetection。

   * 请先安装[MMCV-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)，推荐使用1.4.8 版本。
     ```bash
     pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
     ```
   * ```bash
     cd MCGaze
     pip install -v -e .
     ```

5. 其他可能的问题。

   当你运行代码时有可能会出现这样的报错“FormatCode() got an unexpected keyword argument 'verify”，这是因为yapf包的版本问题，进行以下操作修复。
    ```bash
    pip uninstall yapf
    pip install yapf==0.40.1
    ```
    如果你在复现的过程中遇到了其他的困难，欢迎联系我们。
   
### 准备你的数据集
1. 从[官方网站](http://gaze360.csail.mit.edu/)下载Gaze360数据集。
2. 用我们提供的代码来重新组织数据集排序。你应该首先检查并修改代码中的文件路径，并指定dataset_setting参数的取值（'L2CS' 或对应Gaze360的 'Full'）
   * ```bash
     python tools/gaze360_img_reorganize.py
     ```
3. 从这个[链接]处下载COCO格式的数据集标注。
### 推理及验证

* 运行以下命令以在不同的数据集设置下进行推理和评估。请记住将[数据集路径](./configs/_base_/datasets/gaze360.py)(第三行)更改为您的路径。

##### Gaze360-setting

  ```bash
  bash tools/test_gaze360.sh
  ```

##### l2cs-setting

  ```bash
  bash tools/test_l2cs.sh
  ```



### 从0开始训练

* 执行下面的代码您可以在不同的数据集设置下重新训练模型。
##### Gaze360-setting

  ```bash
  bash tools/train_gaze360.sh
  ```

##### l2cs-setting

  ```bash
  bash tools/train_l2cs.sh
  ```

## 致谢

此代码的灵感来自 [MPEblink](https://github.com/wenzhengzeng/MPEblink),[TeViT](https://github.com/hustvl/TeViT) 和 [MMDetection](https://github.com/open-mmlab/mmdetection)。感谢他们对计算机视觉社区的巨大贡献。

## 引用
即将到来。
