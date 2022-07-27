import os
import numpy as np
import open3d as o3d
import transforms3d
import torch
import time

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField


def find_lib_path(path):
    if os.path.exists(path+'/libPythonInterface.so'):
        return path+'/libPythonInterface.so'
    if os.path.exists(path+'/libPythonInterface.a'):
        return path+'/libPythonInterface.a'
    assert False


def filter_files(folder, exts, data_list, abs=False):
    if not os.path.exists(folder):
        print("Warning: %s not found"%(folder))
        data_list=[]
    else:
        for fileName in os.listdir(folder):
            if os.path.isdir(folder + '/' + fileName):
                filter_files(folder + '/' + fileName, exts, data_list)
            elif fileName.endswith(exts):
                path=folder+'/'+fileName
                if abs:
                    path=os.path.abspath(path)
                data_list.append(path)
    return data_list


def points_transformation(points,transformation):
    # points: n*3, transformation: 4*4
    n=len(points)
    column=np.ones((n,1))
    homogeneous_points=np.concatenate((points,column),axis=1)
    transformed_points=np.matmul(transformation,homogeneous_points.transpose()).transpose()[:,0:3]
    return transformed_points


def normals_transformation(normals, transformation):
    transformed_normals = np.matmul(transformation, normals.transpose()).transpose()
    return transformed_normals


def xyz_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
        Numpy to PointCloud2
        Create a sensor_msgs.PointCloud2 from an array
        of points (x, y, z)
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg


def xyzl_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
        Numpy to PointCloud2
        Create a sensor_msgs.PointCloud2 from an array
        of points (x, y, z, l)
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg


def xyznormal_array_to_pointcloud2(points, normals, stamp=None, frame_id=None):
    '''
        Numpy to PointCloud2
        Create a sensor_msgs PointCloud2 from an array
        of points (x, y, z, nx, ny, nz)
    '''
    points = np.concatenate([points, normals], axis=-1)
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('normal_x', 12, PointField.FLOAT32, 1),
            PointField('normal_y', 16, PointField.FLOAT32, 1),
            PointField('normal_z', 20, PointField.FLOAT32, 1),
        ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = 24 * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg


def trans_like_obj(obj_config, points):
    '''
        Transfer virtual contacts with the object
    '''
    rot = torch.Tensor(transforms3d.quaternions.quat2mat(obj_config[3:7]))
    offset = torch.Tensor(obj_config[:3]).unsqueeze(0).unsqueeze(-1)
    new_points = torch.matmul(rot.T, points - offset)
    return new_points


if __name__ == "__main__":
    pass





