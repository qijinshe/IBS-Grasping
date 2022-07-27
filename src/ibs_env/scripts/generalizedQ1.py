#!/usr/bin/env python
from distanceExact import DistanceExact
from Q1Upperbound import compute_Q1, Directions, ComputeQ1Layer
from hand import Hand, vtk_add_from_hand, vtk_render
from hand_utils import config_from_xml
from utility import trans_like_obj
import numpy as np
import torch
import time
import vtk
import os


def find_bvh_path(shape):
    '''
        Find BVH file path. 
    '''
    dirs = os.path.join(os.getcwd(), 'Grasp_Dataset_v3')
    shape_urdf = os.path.join(dirs, shape, 'gp', "%s.obj.BVH.dat"%(shape))
    return [shape_urdf]
    


def compute_generalized_q1(hand, shape, grasp, dir_num, pens=[], use_center=True, obj_config=None):
    '''
        Refer to [Liu 2020b] for more information
    '''
    m = 1e-3
    M=np.eye(6)
    M[3][3] = m
    M[4][4] = m
    M[5][5] = m
    DistanceExact.mesh_paths = find_bvh_path(shape)
    if len(DistanceExact.mesh_paths) == 0:
        print("No BVH, return 0 as default value")
        default_value = 0
        return default_value
    directions = Directions(res=dir_num)
    M = torch.tensor(M)
    sss = torch.tensor(directions.dirs)
    pss, _, _ = hand.forward(torch.Tensor(grasp.reshape([1,-1])))
    if obj_config is not None:
        print(obj_config)
        pss = trans_like_obj(obj_config, pss)
    dss, onss, pen = DistanceExact.apply(pss)
    pens.append(pen.item())
    if not use_center:
        pss = (pss - pss.mean(dim=-1, keepdim=True))
    with torch.no_grad():
        q1 = ComputeQ1Layer()(M, 0.7, 6.0, pss, dss, onss, sss).item()
        # q1 = ComputeQ1Layer()(M, 0.7, 10, pss, dss, onss, sss).item()
    return q1


def compute_pentration(hand, shape, grasp, obj_config=None):
    '''
        Compute the penetration of the pre-defined visual contacts
    '''
    DistanceExact.mesh_paths = find_bvh_path(shape)
    pss, _, _ = hand.forward(torch.Tensor(grasp.reshape([1,-1])))
    if obj_config is not None:
        pss = trans_like_obj(obj_config, pss)
    _, _, pen = DistanceExact.apply(pss)
    return pen.item()

    
    