import numpy as np


def point_transform(points, translation):
        rot = translation[:3,:3]
        trans = translation[:3,3]

        if len(points.shape) == 2:
            ret = np.einsum( 'ij,aj->ai', rot, points ) + trans
        else:
            ret = np.einsum( 'ij,abj->abi', rot, points ) + trans
        return ret

def depth_to_pointcloud(depth, intrinsic_matrix, cam2world):
    '''
    
    Args:
        depth: np.array [w, h]
        intrinsic_matrix: np.array [3, 3]
        world_to_cam: np.array [4, 4]
    
    Returns:
        pc: np.array [w, h, 3]
    '''
    
    depth = depth.transpose(1, 0)
    w, h = depth.shape

    u0 = intrinsic_matrix[0,2]
    v0 = intrinsic_matrix[1,2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    
    v, u = np.meshgrid( range(h), range(w) )
    z = depth
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy

    z = z.reshape(w, h, 1)
    x = x.reshape(w, h, 1)
    y = y.reshape(w, h, 1)

    depth = depth.transpose(1, 0)
    ret = np.concatenate([x,y,z], axis=-1)
    # translate to world coordinate
    ret = point_transform(ret, cam2world)

    return ret.transpose(1, 0, 2)