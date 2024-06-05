# Stable-Pose
---
Official implementation of Stable-Pose: Leveraging Transformers for
Pose-Guided Text-to-Image Generation.
<p align="center">
  <img src="assets/fig1.png" alt="Figure 1" height="90%" width="90%">
</p>


Stable-Pose is a novel adapter that leverages vision transformers with a coarse-to-fine pose-masked self-attention strategy, specifically designed to efficiently manage precise pose controls during Text-to-Image (T2I) generation. 

The overall structure of Stable-Pose:
<p align="center">
  <img src="assets/framework.png" alt="Overall framework" height="95%" width="95%">
</p>


**Table of Contents**
- [Stable-Pose](#stable-pose)
- [TODO](#todo)
- [Installation](#installation)
- [File structure](#file-structure)
  - [Model and Checkpoints](#model-and-checkpoints)
  - [Data](#data)
  - [Configs](#configs)
  - [Checkpoints and logs](#checkpoints-and-logs)
  - [Evaluation results](#evaluation-results)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [References](#references)

# TODO
- [x] release training codes
- [x] release test codes
- [ ] release scripts for preparing video datasets
- [ ] release trained models
- [ ] add gradio app

# Installation
To get started, first prepare the required environment:
```
# create an environment
conda create -n stable-pose python=3.8.5
# activate the created environment
conda activate stable-pose
# recommended pytorch version
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# install required packages
pip install -r requirements.txt
```
During evaluation, a pretrained pose estimator is needed to predict poses of generated humans. Hence, you might need to install MMPose following this [guide](https://github.com/open-mmlab/mmpose).

# File structure
It's highly recommended to follow our file structure, which allows you to extend the repository to other SOTA techniques easily.
## Model and Checkpoints
We put all the pretrained models we need under 'models' directory:
```
|-- models
    |-- v1-5-pruned.ckpt  # stable diffusion
    |-- higherhrnet_w48_humanart_512x512_udp.pth  # pose esitimator
    |-- init_stable_pose.ckpt  # initialized model for Stable-Pose
```
You may download the Stable Diffusion model from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5) and pose esitimator from [HumanSD](https://github.com/IDEA-Research/HumanSD). 
Note that ```init_stable_pose.ckpt``` is the initialized model for Stable-Pose, you might refer to [Usage](#usage) for details.

## Data
We trained and evaluated on Human-Art [3] and Laion-Human [2], where we follow [HumanSD](https://github.com/IDEA-Research/HumanSD/tree/main) to download and structure the data. Further, we evaluated on three video datasets: UBC Fashion [4], DAVIS [5], and Dance Track [6], where we extracted video frames and built datasets to test (Codes for preparing video datasets will be released soon). The data path is specified in config files.  

## Configs
Config files are structured as:
```
|-- configs
    |-- stable_pose
        |-- humanart.yaml
    |-- mmpose
        |-- ...
```
Stable-Pose and the training/evaluation datasets are configured in the above YAML file, please feel free to make any changes. The 'mmpose' directory contains the config for pose detector, which will be used in evaluation. 

## Checkpoints and logs
Checkpoints and training logs for stable-pose are stored under 'experiments' directory:
```
|-- experiments
    |-- stable_pose
        |-- run_name (specified in config file)
            |-- last.ckpt
            |-- lightning logs
            |-- log_images
            |-- final.pth
```

## Evaluation results
Evaluation results are also stored similarly:
```
|-- outputs
    |-- stable_pose
        |-- run_name (specified in config file)
            |-- metrics (saved metrics as csv files)
            |-- images (generated images)
```

# Usage
We've released training and test codes with sample commands.
Before training, you might need to initialize the Stable-Pose model with weights of Stable Diffusion model, following ControlNet [1]:
```
python prepare_weights.py models/v1-5-pruned.ckpt configs/stable_pose/humanart.yaml models/init_stable_pose.ckpt
```
Training and evluation commands are provided:
```
# please specify config path and initialized model path
python train.py --config configs/stable_pose/humanart.yaml --max_epochs 1 --control_ckpt models/init_stable_pose.ckpt --devices 2 --scale_lr false

# please specify config path and checkpoint path
python eval_pose.py --config_model configs/stable_pose/humanart.yaml --ckpt experiments/stable_pose/run_name/final.pth --scale 7.5
```
For evaluation of image quality (FID, KID), you might need another script:
```
python eval_quality.py
```
Note that for Human-Art, we followed HumanSD and evaluated the quality on each scenario (e.g. cartoon, dance), and the json files are provided in ```val_jsons``` directory.

# Acknowledgments
This repository is built upon [ControlNet](https://github.com/lllyasviel/ControlNet) and [HumanSD](https://github.com/IDEA-Research/HumanSD),  thanks to their great work!

# References
[1] Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding conditional control to text-to-image diffusion models." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2] Ju, Xuan, et al. "HumanSD: A Native Skeleton-Guided Diffusion Model for Human Image Generation." arXiv preprint arXiv:2304.04269 (2023).

[3] Ju, Xuan, et al. "Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[4] Zablotskaia, Polina, et al. "Dwnet: Dense warp-based network for pose-guided human video generation." arXiv preprint arXiv:1910.09139 (2019).

[5] Perazzi, Federico, et al. "A benchmark dataset and evaluation methodology for video object segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[6] Sun, Peize, et al. "Dancetrack: Multi-object tracking in uniform appearance and diverse motion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.