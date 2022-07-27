from hand import Hand,trimesh_to_vtk,write_vtk
import vtk,trimesh,pickle,scipy,json,argparse,random,re
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import os


def config_from_xml(hand,xml_file):
    if not os.path.exists(xml_file):
        return None, None
    tree=ET.parse(xml_file)
    root=tree.getroot()
    robot_dof=root[1][1].text
    robot_dof_parameters=re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",robot_dof)
    pos_np=np.zeros(3,dtype=float)
    rot_np=np.zeros(4,dtype=float)
    joint_np=np.zeros(18, dtype=float)
    for m in range(len(robot_dof_parameters)):
        joint_np[m]=float(robot_dof_parameters[m])
        if abs(joint_np[m])<1e-10:
            joint_np[m]=0
    if hand.use_joint_limit:
        lb,ub=hand.lb_ub()
        joint_np=np.clip(joint_np,lb,ub)
        new_joint_np=np.zeros(hand.nr_dof(),dtype=float)
        for i in range(len(joint_np)):
            if joint_np[i]==ub[i]:
                new_joint_np[i]=-np.log((ub[i]-joint_np[i]+1e-8)/(joint_np[i]-lb[i]))
            elif joint_np[i]==lb[i]:
                new_joint_np[i]=-np.log((ub[i]-joint_np[i])/(joint_np[i]-lb[i]+1e-8))
            else:
                new_joint_np[i]=-np.log((ub[i]-joint_np[i])/(joint_np[i]-lb[i]))
        # def sigmoid(X):
        #     return 1/(1+np.exp(-X))
        # back_joint_np=sigmoid(new_joint_np)*(ub-lb)+lb
        joint_np=new_joint_np
        for i in range(len(joint_np)):
            if np.isinf(joint_np[i]) or np.isnan(joint_np[i]):
                print("DATA ERROR")
                exit()
    #
    robot_transform=root[1][2][0].text
    robot_transform_parameters=re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",robot_transform)
    for m in range(len(robot_transform_parameters)-3):
        rot_np[m]=float(robot_transform_parameters[m])
    pos_np[0]=float(robot_transform_parameters[4])/100.0
    pos_np[1]=float(robot_transform_parameters[5])/100.0
    pos_np[2]=float(robot_transform_parameters[6])/100.0
    hand_config=np.concatenate((np.concatenate((pos_np,rot_np),axis=0),joint_np),axis=0)
    #
    obj_file = root[0][0].text # Corresponding url
    obj_transform=root[0][1][0].text
    obj_transform_parameters=re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",obj_transform)
    # obj_config=np.asarray([None for i in range(7)])
    obj_config=np.asarray([0 for i in range(7)])
    obj_config[0]=float(obj_transform_parameters[4])
    obj_config[1]=float(obj_transform_parameters[5])
    obj_config[2]=float(obj_transform_parameters[6])
    obj_config[3]=float(obj_transform_parameters[0])
    obj_config[4]=float(obj_transform_parameters[1])
    obj_config[5]=float(obj_transform_parameters[2])
    obj_config[6]=float(obj_transform_parameters[3])
    return hand_config,obj_config