# SO-Pose
This repository contains codes of ICCV2021 paper: SO-Pose: Exploiting Self-Occlusion for Direct 6D Pose Estimation
Leveraging self-occlusion we build a novel two-layer representation, better suited for the task of direct 6D pose regression based on 2D-3D correspondences.

Datasets
----------
The code is based on the released code of GDR-Net in this [git](https://github.com/THU-DA-6D-Pose-Group/GDR-Net.git) (The code of GDR-Net is already included)
The struture of the datasets is the same.

Since we need ground truth 2D-3D matching and self-occlusion results, we provide generation methods in .gdrn_selfocc_modeling/tools.
Please refer to generate_*.py.
Note that public renderers (e.g. EGL, GLUMPY) may introduce noise in rendering, the inherent relations between P (2D-3D matching) and Q (self-occlusion) are not guaranteed. So if you use a renderer for efficiency, please make sure that P and Q lie on the same line.

Training and Testing
----------------
Please directly run ./gdrn_selfocc_modeling/main_gdrn.py for training and testing.

Important parameters include
>> config-file : the path to the configuration file.

>> resume: if 'True', continue the training process from the last checkpoint.

>> eval-only: if 'True', directly evalute the model.

Trained Models
--------------
The trained models can be downloaded [here](https://drive.google.com/file/d/136ExcMykxsVVSzOiGQVYspq1fx9Hjd6R/view?usp=sharing).
PLease unzip the trained models in the directory specified in the configuration file.
An example output of the evaluation on LMO is provided.

Citations
--------------
If you find the code useful, please cite the following papers:

@inproceedings{wang2021gdr, \
  title={GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation},\
  author={Wang, Gu and Manhardt, Fabian and Tombari, Federico and Ji, Xiangyang},\
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\
  pages={16611--16621},\
  year={2021}
}

@InProceedings{Di_2021_ICCV,\
    author    = {Di, Yan and Manhardt, Fabian and Wang, Gu and Ji, Xiangyang and Navab, Nassir and Tombari, Federico},\
    title     = {SO-Pose: Exploiting Self-Occlusion for Direct 6D Pose Estimation},\
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},\
    month     = {October},\
    year      = {2021},\
    pages     = {12396-12405}\
}

