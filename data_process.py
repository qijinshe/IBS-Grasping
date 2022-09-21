import os
import xml.dom.minidom as minidom
import numpy as np
import open3d as o3d
from lxml import etree as ET
import numpy as np
import trimesh
import pybullet as p
import os
from tqdm import *

shapes = dict()
grasp_num = 0


def generate_urdf(shape, mesh_dir, urdf_dir, inertial_filename=None):
    urdf_url = os.path.join(urdf_dir, shape.split('.')[0] + '.urdf')
    mesh_url = os.path.join(mesh_dir, shape)
    scale = 10
    obj = ET.Element('robot', attrib={'name': 'object'})
    link = ET.SubElement(obj, 'link', attrib={'concave':'yes', 'name': 'base_link'})
    # Properties
    contact = ET.SubElement(link, 'contact')
    # Frictional properties
    ET.SubElement(contact, 'lateral_friction', attrib={'value':'1.0'})
    ET.SubElement(contact, 'rolling_friction', attrib={'value':'0.0001'})
    ET.SubElement(contact, 'inertia_scaling', attrib={'value':'3.0'})
    v = ET.SubElement(link, 'visual')
    c = ET.SubElement(link, 'collision')
    mesh_url = '{:s}'.format(os.path.abspath(mesh_url))
    # Geometry for visualization
    g = ET.SubElement(v, 'geometry')
    scale_str = (' '.join(['{:.8f}']*3)).format(1/scale, 1/scale, 1/scale)
    ET.SubElement(g, 'mesh',attrib={'filename': mesh_url, 'scale': scale_str})

    # Geometry for collision
    g2 = ET.SubElement(c, 'geometry')
    scale_str = (' '.join(['{:.8f}']*3)).format(1/scale, 1/scale, 1/scale)
    ET.SubElement(g2, 'mesh', attrib={'filename': mesh_url, 'scale': scale_str})

    # Simple material properties of objects
    m = ET.SubElement(v, 'material', attrib={'name': 'white'})
    ET.SubElement(m, 'color', attrib={'rgba': "1 0 0 1"})

    # Inertial
    i = ET.SubElement(link, 'inertial')
    mass_value = float(0.1)
    mass = '{:.8f}'.format(mass_value)  # convert from grams to kg
    ET.SubElement(i, 'mass', attrib={'value': mass})
    if inertial_filename is None:
        # The default value (Pybullet will)
        inertia_tensor = np.array([1, 0, 0, 1, 0, 1])
    if inertia_tensor[0] < 0:
      inertia_tensor *= -1
    inertia_comps = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
    ET.SubElement(i, 'inertia', attrib={k: '{:.4e}'.format(v) for k,v in zip(inertia_comps, inertia_tensor)})

    # The Center of mass
    com = np.array([0.0, 0.0, 0.0])
    com /= scale
    com = (' '.join(['{:.8f}']*3)).format(*com)
    ET.SubElement(i, 'origin', attrib={'xyz': com})

    # Save the file
    tree = ET.ElementTree(obj)
    tree.write(urdf_url, pretty_print=True, xml_declaration=True, encoding='utf-8')


def mesh_process(mesh_url, remove=False):
    binary="utils/mainObjMeshToSDF" # Need to build the project "" first
    if remove:
        if os.path.exists(mesh_url+".BVH.dat"):
            os.remove(mesh_url+".BVH.dat")
        if os.path.exists(mesh_url+".PSet.dat"):
            os.remove(mesh_url+".PSet.dat")
    else:
        if os.path.exists(mesh_url+".PSet.dat") and os.path.getsize(mesh_url+".PSet.dat")>0:
            return
        if os.path.exists(mesh_url+".BVH.dat") and os.path.getsize(mesh_url+".BVH.dat")>0:
            return
    print(mesh_url)
    command=binary+' '+mesh_url+' '+'-0.01'+' '+'0'+' '+'0'
    os.system(command)


def mesh_to_pointcloud(mesh, box_num_rate,  model_savepath,  min_num=5e1, max_num=5e4):
    vertices = np.array(mesh.vertices)
    max_x = vertices[:,0].max()
    max_y = vertices[:,1].max()
    max_z = vertices[:,2].max()
    min_x = vertices[:,0].min()
    min_y = vertices[:,1].min()
    min_z = vertices[:,2].min()
    vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z) # the volume of AABB
    target_num = int( vol*1e6 / box_num_rate) # !!!

    if target_num < min_num:
        target_num = int(min_num)

    if target_num > max_num:
        target_num = int(max_num)

    print( "Vertex Num: %s, Target Num: %s Vol_Num Rate:%d" % (len(vertices), target_num, box_num_rate) )

    pointcloud = mesh.sample_points_poisson_disk(number_of_points = target_num, use_triangle_normal=True)
    pc = np.array(pointcloud.points)
    pointcloud.points = o3d.utility.Vector3dVector(pc)
    ply_path = os.path.join(model_savepath, 'ply')
    pcd_path = os.path.join(model_savepath, 'pcd')
    if not os.path.exists(ply_path):
        os.makedirs(ply_path)
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)
    o3d.io.write_point_cloud( os.path.join(ply_path, "%s.ply" % box_num_rate), pointcloud, write_ascii=True)
    o3d.io.write_point_cloud( os.path.join(pcd_path, "%s.pcd" % box_num_rate), pointcloud, write_ascii=True)


if __name__ == "__main__":
    p.connect(p.DIRECT)
    grasp_dir = 'Grasp_Dataset/grasps'
    model_dir = 'Grasp_Dataset/good_shapes'
    new_root_dir = 'Grasp_Dataset_v4'
    
    if not os.path.exists(new_root_dir):
        os.mkdir(new_root_dir)
    grasp_list = os.listdir(grasp_dir)
    model_list = os.listdir(model_dir)
    remove_list = []
    with open('remove','r') as f:
       remove_list = [line[:-1] for line in f.readlines()]
    
    for obj in model_list:
        str_list = obj.split('.')
        name = str_list[0]
        if name in remove_list:
            print(name)
            continue
        shapes[name] = 0
        old_path = os.path.join(model_dir, name)
        new_path = os.path.join(new_root_dir, name)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            os.mkdir(os.path.join(new_path, 'shape'))
            os.mkdir(os.path.join(new_path, 'grasps'))
        file_type = str_list[-1]
        # if file_type == 'off':
        #     mesh = trimesh.load(os.path.join(model_dir, obj), process=False)
        #     mesh.apply_scale(1/100)
        #     mesh.export(os.path.join(new_path, 'shape', name+'.ply'), file_type='ply')
        #     mesh.export(os.path.join(new_path, 'shape', name+'.obj'), file_type='obj')
        #     mesh = o3d.io.read_triangle_mesh(os.path.join(new_path, 'shape', name+'.ply'))
        #     mesh = mesh.compute_vertex_normals()
        #     o3d.io.write_triangle_mesh(os.path.join(new_path, 'shape', name+'.ply'), mesh)
        # else:
        #     old_file_path = os.path.join(model_dir, obj)
        #     new_file_path = os.path.join(new_path, 'shape')
        #     os.system("cp %s %s"  % (old_file_path, new_file_path))

    # Generate Pointcloud
    # for name in shapes.keys():
    #     rates = [200, 400, 600, 800, 1000]
    #     model_savepath = os.path.join(new_root_dir, name)
    #     model_url = os.path.join(new_root_dir, name, 'shape', name+'.ply')
    #     mesh = o3d.io.read_triangle_mesh(model_url)
    #     for box_num_rate in rates:
    #         mesh_to_pointcloud(mesh, box_num_rate, model_savepath)
    #         print(model_savepath)
    
    # # Generate VHACD object and URDF
    # for name in shapes.keys():
    #     urdf_dir = os.path.join(new_root_dir, name, "urdf")
    #     if not os.path.exists(urdf_dir):
    #         os.makedirs(urdf_dir)
    #     vhacd_url = os.path.join(new_root_dir, name, "urdf",  "%s_vhacd.obj"%(name))
    #     log_url = os.path.join(new_root_dir, name, "urdf",  "%s_vhacd.log"%(name))
    #     model_url = os.path.join(new_root_dir, name, 'shape', name+'.obj')
    #     p.vhacd(model_url, vhacd_url, log_url)
    #     generate_urdf("%s_vhacd.obj"%(name), urdf_dir, urdf_dir)

    # Generate BVH and PSet
    # for name in shapes.keys():
    #     gp_dir = os.path.join(new_root_dir, name, 'gp')
    #     # os.system('rm -rf %s'%(gp_dir))
    #     # if not name[:3] == 'ycb':
    #     #     print("Jump")
    #     #     continue
    #     if not os.path.exists(gp_dir):
    #         os.makedirs(gp_dir)
    #     model_url = os.path.join(new_root_dir, name, 'shape', name+'.obj')
    #     os.system("cp %s %s"  % (model_url, gp_dir))
    #     model_url2 = os.path.join(new_root_dir, name, 'gp', name+'.obj')
    #     # print(model_url2)
    #     mesh_process(model_url2)
        
    # Move grasp file to the directory
    # for obj in tqdm(grasp_list):
    #     gp = os.path.join(grasp_dir, obj)
    #     name = obj.split('.')[0]
    #     grasps = os.listdir(gp)
    #     min_no = 100
    #     best_grasp = None
    #     for gg in grasps:
    #         file_name = gg.split('.')[0]
    #         if file_name[:5] != 'grasp':
    #             continue
    #         no = int(file_name[5:])
    #         if no < min_no:
    #             best_grasp = gg
    #             min_no = no
    #     if best_grasp is None:
    #         continue
    #     xml_path = os.path.join(gp, best_grasp)
    #     tree=ET.parse(xml_path)
    #     root=tree.getroot()
    #     model_url = root[0][0].text
    #     name = (model_url.split('/')[-1]).split('.')[0]
    #     if name in shapes:
    #         shapes[name] += 1
    #         new_name = "grasp_%d.xml" %(shapes[name])
    #         new_path = os.path.join(new_root_dir, name, 'grasps', new_name)
    #         tree.write(new_path, encoding="utf-8", xml_declaration=True)
