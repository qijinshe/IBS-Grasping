
This project is based on our SIGGRAPH2022 paper, [Learning High-DOF Reaching-and-Grasping via Dynamic
Representation of Gripper-Object Interaction](https://arxiv.org/abs/2204.13998)


## Introduction
In this project, We adopt Interaction Bisector Surface (IBS), which is a surface composed of points at equal distances to two close by objects as an observation representation in learning high-DOF reach-and-grasp planning. We found that IBS is able and effective to guide the motion of the gripper.


## Installation
This probject partly depends on ROS, you need to [install ROS](http://wiki.ros.org/melodic/Installation) first (In our environment, we use ROS melodic).

The necessary Python packages are list in the file "requirement.txt".


You can run the command to install them all:
> pip install -r requirement.txt

"deepdifferentiablegrasp" is also important in our project.
Here we only provide [compiled files](https://drive.google.com/drive/folders/1xSJMm63BTmK7nC0Cjk6wA3so6irjcQXJ).
Note that specific versions of 'Boost'(1.58) and 'Mosek'(9.0) are necessary to run the file.

Last, you need to compile ROS packages in the root directory:
>catkin_make


## Running
First, run devel/setup.bash to configurate environment setting:
> source devel/setup.bash


Before test or training, you need to run IBS service:
> rosrun ibs_grasping ibs_env


For test:
>rosrun ibs_env main.py --model_name [model_name]


For quick test:
>rosrun ibs_env main.py --model_name [model_name] --quick


For trainning
>rosrun ibs_env main.py --train_model


[Pretrained models](https://drive.google.com/drive/folders/1fK5Ni-Zc8-WNkAGDv7tapAMoyKSDmpja) are also provided


# Visualization
You can use [RVIZ](http://wiki.ros.org/rviz) package to visualize the IBS. The configuration file are provided in the root directory (ibs_visualization.rviz).


## Data Preparation
The objects used in this work can be download [here](https://drive.google.com/file/d/1LD12JHqKKb4TjTy4vGZLrJoXsQZLOBPv/view),
which includes 500 watertight objects collected from four datasets (KIT, GD, YCB, BIGBIRD) as well as their feasible grasps


The data processing script is provided in the root directory.
The script will remove objects unsuitable for grasping and generate following files for you:
* ".pcd": pointclouds sampled from origin meshes, used for IBS computation.
* "_vhacd.obj" and ".urdf": approximate convex decomposition of objectsusing the VHACD algorithm and the URDF wrappers of objects, used for Pybullet Simulation.
* ".bvh" (optional): precomputed files, used for grasp quality computation. If you don't want to train models or compute grasp quality, you don't need to generate these files.


## Acknowledgments
Some codes are modified from some projects. Thanks for their
excellent works: 


[pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic/)


[AdaGrasp](https://github.com/columbia-ai-robotics/adagrasp)


deepdifferentiablegrasp (see "contact")


## Citation
If you are interested in this work, please cite the following paper:

```
@article {she_sig22,
    title = {Learning High-DOF Reaching-and-Grasping via Dynamic Representation of Gripper-Object Interaction},
    author = {Qijin She and Ruizhen Hu and Juzhan Xu and Min Liu and Kai Xu and Hui Huang},
    journal = {ACM Transactions on Graphics (SIGGRAPH 2022)},
    volume = {41},
    number = {4},
    year = {2022}
}
```


## License
The source code is released under GPLv3 license.


## Contact
If you have any questions, feel free to email Qijin She (qijinshe@outlook.com)


If you want to get complete code of "deepdifferentiablegrasp", please contact Liu Min (gfsliumin@gmail.com)
