# NeRFlect - Reflecting NeRFs as Dataset Generators
Extend the horizons of Neural Radiance Fields with NeRFlect, an extension to various NeRF models. This repository only includes the enhanced [HumanRF](https://github.com/synthesiaresearch/humanrf) framework within NeRFlect, and it demonstrates the improvements for automated rendering of novel views of human actors.
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Potential Applications](#potential-applications)
- [Report](#report)
- [Installation](#installation)
- [Detailed User Guide](#detailed-user-guide)
- [Acknowledgment](#acknowledgment)
  
## Introduction
NeRFlect enhances the capabilities of the HumanRF model, allowing users to generate images of human actors in the desired viewpoints. One may select to orbit around the object starting from a selected camera or sample new cameras uniformly on a sphere surface to generate new images from different perspectives of the object in the 3D world.

## Features
- **Orbit Sampling:**  Rotate the camera 360 degrees around your scene and specify the number of samples you want.
- **Sphere Surface Sampling:** Define your viewpoint and sample uniformly with a radius of your choice on a sphere surface.
- **High-Fidelity Visuals:** Built upon the robustness of NeRF models, expect high-quality renderings.
  
## Potential Applications
- **Augmented Dataset Generation:** Enhance AI-driven human visual datasets by simulating diverse camera angles and perspectives using NeRFlect, promoting the training of more robust models.
- **Avatar Creation for Virtual Worlds:** Automate personalized avatar generation for VR and AR platforms, offering users a comprehensive 360-degree 3D view of their digital existence.

## Report

For an in-depth exploration of the project, including methodologies, experiments, and results, please refer to the [Project Report](report/NeRFlect.pdf).

## Installation

```bash

git clone --depth=1 --recursive https://github.com/korhankemalkaya/NeRFlect.git

# Install GLM
sudo apt-get install libglm-dev

# Install required packages and Tiny CUDA NN.
pip install -r requirements.txt
pip install pandas #Additional to the original HumanRF environment
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install ActorsHQ package (dataset and data loader)
cd actorshq
pip3 install .

# Install HumanRF package (method)
cd ../humanrf
pip3 install .

# Add the installation folder to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/repo
```

## Detailed User Guide

To start, one should download a part of ActorsHQ, which can be done by the following command:

```bash
./actorshq/dataset/download_manager.py \
    actorshq_access_4x.yaml \
    /tmp/actorshq \
    --actor Actor01 \
    --sequence Sequence1 \
    --scale 4 \
    --frame_start 15 \
    --frame_stop 65
```
For that, you will need an access file, `actorshq_access_4x.yaml`, which you can request [here](https://www.actors-hq.com/), see [original HumanRF repository](https://github.com/synthesiaresearch/humanrf)  for more details.

 Having access to ActorsHQ, one can run the program by specifying some vital arguments:

   Example use can be found below:
```bash
./humanrf/run.py \
    --config example_humanrf \
    --workspace /tmp/example_workspace \
    --dataset.path /tmp/actorshq \
```
 - **camera_parameters:** *An optional configuration file argument. It should be specified according to the calibration data format, whose example can be found below. It allows one to enter a camera's parameter manually.*
 - **"test.trajectory_via_calibration_file** *A configuration file argument. If the camera_parameters are not specified, then the user is expected to pass the path of the calibration file to this argument.* 
 - **is_orbited:** *A configuration file argument. If True, the model rotates the input camera in a circular orbit and samples new cameras.*
 - **is_uniformed:** *A configuration file argument. If True, the model performs a uniform (evenly distributed) sampling of new cameras on a sphere surface*
 - **radius:** *An optional configuration file argument. If either is_uniformed or is_orbited is True, the user can specify the sphere's radius. If not specified, the radius is calculated from the given camera and the object center.*
 - **sample_number:** *A configuration file argument for the sampling count.*
 - **specific_frame:** *An optional configuration file argument to render from a specific frame. If not specified, HumanRF's original framing process is used.*
 - **object_center_addition:** *An optional configuration file argument. It allows users to manipulate the object center. Assuming the original object center is calculated as (a,b,c), one can specify this parameter as (x,y,z) to render images around the new center (a+x, b+y, c+z).*
 - **test.checkpoint:** *A configuration file argument for the path of the trained model.*

   
__Important Notices:__
- In the uniform sampling process, the number of outputs may not match the sample_number argument due to the truncation of the floating point numbers when converting to integers.
- You can find an example of the calibration data format below:
```
name, w, h, rx, ry, rz, tx, ty, tz, fx, fy, px, py
Cam001, 4112, 3008, 3.14159265359, 0.0, 0.0, 1.0, 0.0, 230.0, 1.773863, 1.773863, 0.5, 0.5
...
```

Here, the rotation vector `rx, ry, rz` is in axis-angle format, and the focal length and principal point are normalized by the image width and height.
## Acknowledgment
Heartfelt gratitude to the [HumanRF team](https://github.com/synthesiaresearch/humanrf) for their foundational work.
