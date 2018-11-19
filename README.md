# chainer-pose-proposal-net

- This is an implementation of [Pose Proposal Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sekii_Pose_Proposal_Networks_ECCV_2018_paper.pdf) with Chainer including training and prediction tools.

# License

Copyright (c) 2018 Idein Inc. & Aisin Seiki Co., Ltd.
All rights reserved.

This project is licensed under the terms of the [license](LICENSE).

# Training

- Prior to training, let's download dataset. You can train with MPII or COCO dataset by yourself.
- For simplicity, we will use docker image of [idein/chainer](https://hub.docker.com/r/idein/chainer/) which includes Chainer, ChainerCV and other utilities with CUDA driver. This will save time setting development environment.

## Prepare Dataset

### MPII

- If you train with COCO dataset you can skip.
- Access [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/) and jump to `Download` page. Then download and extract both `Images (12.9 GB)` and `Annotations (12.5 MB)`.

#### Create `mpii.json`

We need decode `mpii_human_pose_v1_u12_1.mat` to generate `mpii.json`. This will be used on training or evaluating test dataset of MPII.

```
$ sudo docker run --rm -v $(pwd):/work -v path/to/dataset:/data -w /work idein/chainer:4.5.0 python3 convert_mpii_dataset.py /data/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat /data/mpii.json
```

It will generate `mpii.json` at `path/to/dataset` where is the root directory of MPII dataset. For those who hesitate to use Docker, you may edit `config.ini` as necessary.

### COCO

- If you train with MPII dataset you can skip.
- Access [COCO dataset](http://cocodataset.org/) and jump to `Dataset` -> `download`. Then download and extract `2017 Train images [118K/18GB]`, `2017 Val images [5K/1GB]` and `2017 Train/Val annotations [241MB]`.

## Running Training Scripts

```
$ sudo docker run --rm -v $(pwd):/work -v path/to/dataset:/data -w /work idein/chainer:4.5.0 python3 train.py
```

- Optional argument `--runtime=nvidia` maybe require for some environment.
- This will train a model the base network is MobileNetV2 with MPII dataset located in `path/to/dataset` on host machine.
- If we would like to train with COCO dataset, edit a part of `config.ini` as follow:

before

```
# parts of config.ini
[dataset]
type = mpii
```

after

```
# parts of config.ini
[dataset]
type = coco
```

- We can choice ResNet based network as the original paper adopts. Edit a part of `config.ini` as follow:

before

```
[model_param]
model_name = mv2
```

after

```
[model_param]
# you may also choice resnet34 and resnet50
model_name = resnet18
```

# Prediction

- Very easy, all we have to do is:

```
$ sudo docker run --rm -v $(pwd):/work -v path/to/dataset:/data -w /work idein/chainer:4.5.0 python3 predict.py
```

# Demo: Realtime Pose Estimation

We tested on an Ubuntu 16.04 machine with GPU GTX1080(Ti)

## Build Docker Image for Demo

We will build OpenCV from source to visualize the result on GUI.

```
$ cd docker/gpu
$ cat build.sh
docker build -t ppn .
$ sudo bash build.sh
```

## Run video.py

- Set your USB camera that can recognize from OpenCV.

- Run `video.py`

```
$ python video.py
```

or

```
$ sudo bash run_video.sh
```

## High Performance Version
- To use feature of [Static Subgraph Optimizations](http://docs.chainer.org/en/stable/reference/static_graph_design.html) to accelerate inference speed, we should install Chainer 5.0.0 and CuPy 5.0.0 .
- Prepare high performance USB camera so that takes more than 60 FPS.
- Run `high_speed.py` instead of `video.py`
- Do not fall from the chair with surprise :D.

# Appendix

- [Implementation of Pose Proposal Networks (NotePC with e-GPU)](https://twitter.com/IdeinInc/status/1059385580180500482)
- [Demo: Pose Proposal Network on a Raspberry Pi](https://www.youtube.com/watch?v=L_kAUnAgkfg)
  - It runs on Raspberry Pi 3 locally using its GPU (VideoCore IV) with almost 10 FPS.
  - [It also runs on Raspberry Pi Zero](https://twitter.com/9_ties/status/1059750417679114240) with 6.6 FPS.

# Citation
Please cite the paper in your publications if it helps your research:

    @InProceedings{Sekii_2018_ECCV,
      author = {Sekii, Taiki},
      title = {Pose Proposal Networks},
      booktitle = {The European Conference on Computer Vision (ECCV)},
      month = {September},
      year = {2018}
      }
