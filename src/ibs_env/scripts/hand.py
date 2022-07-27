import os,trimesh,trimesh.creation,copy,math,re,pickle,shutil,vtk,scipy,torch,transforms3d
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from utility import points_transformation, normals_transformation

# import kornia
# torch.set_default_dtype(torch.float64)

import time

def normalize_quaternion(quaternion: torch.Tensor,
                         eps: float = 1e-12) -> torch.Tensor:
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    return F.normalize(quaternion, p=2, dim=-1, eps=eps)

def Quat2mat(quaternion): 
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)
    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)
    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.0)
    matrix: torch.Tensor = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy)
    ], dim=-1).view(-1, 3, 3)
    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix

def DH2trans(theta, d, r, alpha):
    Z=np.asarray([[math.cos(theta),-math.sin(theta),0,0],
                  [math.sin(theta), math.cos(theta),0,0],
                  [              0,               0,1,d],
                  [              0,               0,0,1]])
    X=np.asarray([[1,              0,               0,r],
                  [0,math.cos(alpha),-math.sin(alpha),0],
                  [0,math.sin(alpha), math.cos(alpha),0],
                  [0,              0,               0,1]]) 
    tr=np.matmul(Z,X)
    return tr,Z,X

ZC=torch.tensor([[1.,0, 0, 0],
                [0, 1.,0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

ZS=torch.tensor([[0,-1.,0, 0],
                [1.,0, 0, 0],  
                [0, 0, 0, 0],
                [0, 0, 0, 0]])
                
def DH2trans_torch(theta, d, r, alpha):
    Zc = ZC.type(theta.type())
    Zs = ZS.type(theta.type())
    Z0=torch.tensor([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 1.,d],
                     [0, 0, 0, 1.]]).type(theta.type())
    Z =  torch.cos(theta).view([-1,1,1])*Zc
    Z += torch.sin(theta).view([-1,1,1])*Zs
    Z += Z0
    X=torch.tensor([[1,              0,               0,r],
                    [0,math.cos(alpha),-math.sin(alpha),0],
                    [0,math.sin(alpha), math.cos(alpha),0],
                    [0,              0,               0,1]]).type(theta.type())
    return torch.matmul(Z,X),Z,X

def trimesh_to_vtk(trimesh):
    r"""Return a `vtkPolyData` representation of a :map:`TriMesh` instance
    Parameters
    ----------
    trimesh : :map:`TriMesh`
        The menpo :map:`TriMesh` object that needs to be converted to a
        `vtkPolyData`
    Returns
    -------
    `vtk_mesh` : `vtkPolyData`
        A VTK mesh representation of the Menpo :map:`TriMesh` data
    Raises
    ------
    ValueError:
        If the input trimesh is not 3D.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    # if trimesh.n_dims != 3:
    #     raise ValueError('trimesh_to_vtk() only works on 3D TriMesh instances')

    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    trimesh.vertices = np.asarray(trimesh.vertices)
    points.SetData(numpy_to_vtk(trimesh.vertices, deep=0, array_type=vtk.VTK_FLOAT))
    mesh.SetPoints(points)

    cells = vtk.vtkCellArray()

    # Seemingly, VTK may be compiled as 32 bit or 64 bit.
    # We need to make sure that we convert the trilist to the correct dtype
    # based on this. See numpy_to_vtkIdTypeArray() for details.
    isize = vtk.vtkIdTypeArray().GetDataTypeSize()
    req_dtype = np.int32 if isize == 4 else np.int64
    cells.SetCells(trimesh.faces.shape[0],
                   numpy_to_vtkIdTypeArray(
                       np.hstack((np.ones(trimesh.faces.shape[0])[:, None] * 3,
                                  trimesh.faces)).astype(req_dtype).ravel(),
                       deep=1))
    mesh.SetPolys(cells)
    return mesh

def write_vtk(polydata, name):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(name)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

def points_to_vtk(points):
    contacts=None
    for i in range(len(points)):
        point_mesh=trimesh.primitives.Sphere(radius=0.025,center=np.asarray([points[i][0], points[i][1],points[i][2]]))
        if contacts==None:
            contacts=point_mesh
        else:
            contacts+=point_mesh
    return trimesh_to_vtk(contacts)

class Link:
    def __init__(self,mesh,mesh_id,parent,transform,dhParams):
        self.mesh=mesh
        self.mesh_id=mesh_id
        self.parent=parent
        self.transform=transform
        self.transform_torch=torch.tensor(self.transform[0:3,:])
        self.dhParams=dhParams  #id, mul, trans, d, r, alpha, min, max
        self.joint_transform=None
        self.joint_rotation=None #0221
        self.end_effector=[]
        self.children=[]
        if parent is not None:
            parent.children.append(self)
        # Add Label
        self.label = 0
        self.part_label = 0

    def convert_to_revolute(self):
        if self.dhParams and len(self.dhParams)>1:
            tmp=self.children
            self.children=[]
            #create middle
            middle=Link(self.mesh,self.mesh_id,self,transforms3d.affines.compose(np.zeros(3),np.eye(3, 3),[1,1,1]),[self.dhParams[1]])
            middle.dhParams=[self.dhParams[1]]
            middle.children=tmp
            for c in tmp:
                c.parent=middle
            #update self
            self.dhParams=[self.dhParams[0]]
            self.mesh=None
        for c in self.children:
            c.convert_to_revolute()
    
    def convert_to_zero_mean(self):
        if self.dhParams:
            mean=(self.dhParams[0][6]+self.dhParams[0][7])/2
            rot=transforms3d.axangles.axangle2aff([0, 0, 1], mean, None)
            self.transform=np.matmul(self.transform,rot)
            self.dhParams[0][6]-=mean
            self.dhParams[0][7]-=mean
        for c in self.children:
            c.convert_to_zero_mean()
    
    def max_dof_id(self):
        ret=0
        if self.dhParams:
            for dh in self.dhParams:
                ret=max(ret,dh[0])
        if self.children:
            for c in self.children:
                ret=max(ret,c.max_dof_id())
        return ret

    def lb_ub(self,lb,ub):
        if self.dhParams:
            for dh in self.dhParams:
                lb[dh[0]]=dh[6]
                ub[dh[0]]=dh[7]
        if self.children:
            for c in self.children:
                c.lb_ub(lb,ub)

    def forward_kinematics(self,root_trans,dofs):
        if not self.dhParams:
            self.joint_transform = root_trans
            self.joint_rotation = root_trans[:3,:3] #
        else:
            self.joint_transform=np.matmul(self.parent.joint_transform, self.transform)
            self.joint_rotation = self.joint_transform[:3,:3] #
            for dh in self.dhParams: # D-H
                theta=float(dofs[dh[0]])
                theta=max(theta,dh[6])
                theta=min(theta,dh[7])
                dh,Z,X=DH2trans(theta*dh[1]+dh[2],dh[3],dh[4],dh[5])
                self.joint_transform = np.matmul(self.joint_transform,dh)
                self.joint_rotation = self.joint_transform[:3,:3] #e
        if self.children:
            for c in self.children:
                c.forward_kinematics(root_trans,dofs)
    
    def forward(self,root_trans,dofs):
        if not self.dhParams:
            self.joint_transform_torch=root_trans
            jr,jt=torch.split(root_trans,[3,1],dim=2)
        else:
            pr,pt=torch.split(self.parent.joint_transform_torch,[3,1],dim=2)
            r,t=torch.split(self.transform_torch,[3,1],dim=1)
            jr=torch.matmul(pr,r.type(pr.type()))
            jt=torch.matmul(pr,t.type(pr.type()))+pt
            for dh in self.dhParams:
                _,theta,_=torch.split(dofs,[dh[0],1,dofs.shape[1]-dh[0]-1],dim=1)
                theta=torch.clamp(theta,min=dh[6],max=dh[7])
                dh,_,_=DH2trans_torch(theta*dh[1]+dh[2],dh[3],dh[4],dh[5])
                dh,_=torch.split(dh,[3,1],dim=1)
                dhr,dht=torch.split(dh,[3,1],dim=2)
                #cat
                jt=torch.matmul(jr,dht)+jt
                jr=torch.matmul(jr,dhr)
            self.joint_transform_torch=torch.cat([jr,jt],dim=2)
        #compute ret
        if len(self.end_effector)==0:
            retv=None
            retn=None
        else:
            eep=[]
            een=[]
            for ee in self.end_effector:
                eep.append(ee[0].tolist())
                een.append(ee[1].tolist())
            eep=torch.transpose(torch.tensor(eep),0,1).type(jt.type())
            een=torch.transpose(torch.tensor(een),0,1).type(jt.type())
            retv=torch.matmul(jr,eep)+jt    # Position
            retn=torch.matmul(jr,een)       # Normal
        rett=[self.joint_transform_torch]
        # descend
        if self.children:
            for c in self.children:
                cv,cn,ct=c.forward(root_trans,dofs)
                if retv is None:
                    retv=cv
                    retn=cn
                elif cv is not None:
                    retv=torch.cat([retv,cv],dim=2)
                    retn=torch.cat([retn,cn],dim=2)
                rett+=ct
        return retv,retn,rett

    def forward_sampled(self,root_trans,dofs):
        if not self.dhParams:
            self.joint_transform_torch=root_trans
            jr,jt=torch.split(root_trans,[3,1],dim=2)
        else:
            pr,pt=torch.split(self.parent.joint_transform_torch,[3,1],dim=2)
            r,t=torch.split(self.transform_torch,[3,1],dim=1)
            jr=torch.matmul(pr,r.type(pr.type()))
            jt=torch.matmul(pr,t.type(pr.type()))+pt
            for dh in self.dhParams:
                _,theta,_=torch.split(dofs,[dh[0],1,dofs.shape[1]-dh[0]-1],dim=1)
                theta=torch.clamp(theta,min=dh[6],max=dh[7])
                dh,_,_=DH2trans_torch(theta*dh[1]+dh[2],dh[3],dh[4],dh[5])
                dh,_=torch.split(dh,[3,1],dim=1) 
                dhr,dht=torch.split(dh,[3,1],dim=2)
                #cat
                jt=torch.matmul(jr,dht)+jt
                jr=torch.matmul(jr,dhr)
            self.joint_transform_torch=torch.cat([jr,jt],dim=2)
        #compute ret
        if len(self.sample_points)==0:
            retv=None
        else:
            eep=torch.transpose(torch.tensor(self.sample_points),0,1).type(jt.type())
            retv=torch.matmul(jr,eep)+jt
        # descend
        if self.children:
            for c in self.children:
                cv =c.forward_sampled(root_trans,dofs)
                if retv is None:
                    retv=cv
                elif cv is not None:
                    retv=torch.cat([retv,cv],dim=2)
        return retv
        
    def draw(self):
        if self.mesh:
            ret=copy.deepcopy(self.mesh).apply_transform(self.joint_transform)
        else: ret=None
        if self.children:
            for c in self.children:
                if ret:
                    ret+=c.draw() 
                else: ret=c.draw()
        return ret

    def add_area(self):
        if self.mesh:
            area=self.mesh.area
        else: area=None
        if self.children:
            for c in self.children:
                if area:
                    area+=c.add_area()
                else: area=c.add_area()
        return area

    def sample(self,count_over_area,path,re_sample=False):
        if self.mesh:
            if re_sample:
                # self.sample_points=trimesh.sample.sample_surface_even(self.mesh,int(self.mesh.area*count_over_area))[0]
                self.sample_points, sample_idx=trimesh.sample.sample_surface_even(self.mesh,int(self.mesh.area*count_over_area))
                self.sample_normals = self.mesh.face_normals[sample_idx]
                np.save(path+self.mesh_id,self.sample_points)
                np.save(path+self.mesh_id+'_normal',self.sample_normals)
            else:
                if os.path.exists(path+self.mesh_id+'.npy') and os.path.exists(path+self.mesh_id+'_normal.npy'): # Read Saved sampled points
                    self.sample_points = np.load(path+self.mesh_id+'.npy')
                    self.sample_normals = np.load(path+self.mesh_id+'_normal'+'.npy')
                else:
                    # Sampled points on mesh
                    # self.sample_points=trimesh.sample.sample_surface_even(self.mesh,int(self.mesh.area*count_over_area))[0]
                    self.sample_points, sample_idx=trimesh.sample.sample_surface_even(self.mesh,int(self.mesh.area*count_over_area))
                    self.sample_normals = self.mesh.face_normals[sample_idx]
                    np.save(path+self.mesh_id, self.sample_points)
                    np.save(path+self.mesh_id+'_normal', self.sample_normals)
        else: 
            self.sample_points=None
        if self.children:
            for c in self.children:
                c.sample(count_over_area,path,re_sample)
    
    def sample_fwk(self):
        # print(self.label)
        if self.sample_points is not None:
            sample_points_transformed=points_transformation(self.sample_points,self.joint_transform)
        else: 
            sample_points_transformed=None
        if self.children:
            for c in self.children:
                if sample_points_transformed is not None:
                    sample_points_transformed=np.concatenate((sample_points_transformed,c.sample_fwk()))
                else: 
                    sample_points_transformed=c.sample_fwk()
        return sample_points_transformed

    def sample_fwk_normal(self):
        if self.sample_points is not None:
            normal = self.sample_normals.copy().T
            sample_normals_transformed= np.matmul(self.joint_rotation, normal).T
        else: 
            sample_normals_transformed=None
        if self.children:
            for c in self.children:
                if sample_normals_transformed is not None:
                    sample_normals_transformed=np.concatenate((sample_normals_transformed,c.sample_fwk_normal()))
                else: 
                    sample_normals_transformed=c.sample_fwk_normal()
        return sample_normals_transformed

    def get_internal_points(self):
        if self.sample_normals is not None:
            normal = np.array([0, 0, 1])
            if self.part_label in [11, 12, 13, 14,
                                     20, 21, 22, 23,
                                     30, 31, 32, 33,
                                     40, 41, 42, 43,
                                     51, 52]:
                normal = np.array([0, 1, 0])
            # The value indicating which side of the gripper [-1,1]
            internal_labels = np.matmul(self.sample_normals, normal)
        else: 
            internal_labels=None
        if self.children:
            for c in self.children:
                if internal_labels is not None:
                    internal_labels = np.concatenate((internal_labels, c.get_internal_points()))
                else: 
                    internal_labels = c.get_internal_points()
        return internal_labels

    def get_labels(self):
        '''
        '''
        if self.sample_points is not None:
            labels = np.ones(self.sample_points.shape[0]) * self.label
        else: 
            labels = None
        if self.children:
            for c in self.children:
                if labels is not None:
                    labels=np.concatenate((labels,c.get_labels()))
                else: 
                    labels=c.get_labels()
        return labels

    def get_part_labels(self):
        '''
        '''
        if self.sample_points is not None:
            labels = np.ones(self.sample_points.shape[0]) * self.part_label
        else: 
            labels = None
        if self.children:
            for c in self.children:
                if labels is not None:
                    labels=np.concatenate((labels,c.get_part_labels()))
                else: 
                    labels=c.get_part_labels()
        return labels

    def Rx(self):
        tr,Z,X=DH2trans(0,self.dhParams[0][3],self.dhParams[0][4],self.dhParams[0][5])
        return X[0:3,0:3]
    
    def Rt(self):
        return self.transform[0:3,0:3]
    
    def tz(self):
        tr,Z,X=DH2trans(0,self.dhParams[0][3],self.dhParams[0][4],self.dhParams[0][5])
        return Z[0:3,3]
    
    def tx(self):
        tr,Z,X=DH2trans(0,self.dhParams[0][3],self.dhParams[0][4],self.dhParams[0][5])
        return X[0:3,3]
    
    def tt(self):
        return self.transform[0:3,3]
    
    def R(self):
        return self.joint_transform[0:3,0:3]
        
    def t(self):
        return self.joint_transform[0:3,3]
    
    def add_end_effector(self,location,normal):
        self.end_effector.append([location,normal])
    
    def get_end_effector(self):
        return self.end_effector
    
    def get_end_effector_all(self):
        ret=[]
        for ee in self.end_effector:
            loc=np.matmul(self.joint_transform[0:3,0:3],ee[0])
            loc=np.add(loc,self.joint_transform[0:3,3])
            nor=np.matmul(self.joint_transform[0:3,0:3],ee[1])
            ret.append([loc.tolist(),nor.tolist()])
        if self.children:
            for c in self.children:
                ret+=c.get_end_effector_all()
        return ret

class Hand(torch.nn.Module):
    def __init__(self,hand_path,scale,use_joint_limit=True,use_quat=True,use_eigen=False):
        super(Hand,self).__init__()
        self.build_tensors()
        self.hand_path=hand_path
        self.use_joint_limit=use_joint_limit
        self.use_quat=use_quat
        self.use_eigen=use_eigen
        self.eg_num=0
        if self.use_quat:
            self.extrinsic_size=7
        else: self.extrinsic_size=6
        self.contacts=self.load_contacts(scale)
        self.tree=ET.parse(self.hand_path+'hand.xml')
        self.root=self.tree.getroot()
        self.labels = None
        self.part_labels = None
        #load other mesh
        self.linkMesh={}
        for file in os.listdir(self.hand_path+'/off'):
            if file.endswith('.off'):
                name=file[0:len(file)-4]
                self.linkMesh[name]=trimesh.load_mesh(self.hand_path+'/off/'+file).apply_scale(scale)
                # mesh = o3d.io.read_triangle_mesh(os.path.join(self.hand_path+'/off/'+file))
                # mesh = mesh.compute_vertex_normals()
                # o3d.visualization.draw_geometries([mesh])
                # o3d.io.write_triangle_mesh(os.path.join(self.hand_path+'/off/'+file[:-4]+'.ply'), mesh)
                
        #build links
        transform = transforms3d.affines.compose(np.zeros(3), np.eye(3, 3), [1,1,1])
        self.palm=Link(self.linkMesh[self.root[0].text[:-4]],self.root[0].text[:-4],None,transform,None)
        self.part_label = 0 + 0
        for i in range(len(self.contacts[-1,0])):
            self.palm.add_end_effector(self.contacts[-1,0][i][0],self.contacts[-1,0][i][1])
        chain_index=0
        for chain in self.root.iter('chain'):
            #load chain
            transform = transforms3d.affines.compose(np.zeros(3), np.eye(3, 3), [1,1,1])
            for i in range(len(chain[0])):
                if chain[0][i].tag == 'translation':
                    translation = re.findall(r"\-*\d+\.?\d*", chain[0][i].text)
                    trans = np.zeros(3)
                    trans[0] = float(translation[0])*scale
                    trans[1] = float(translation[1])*scale
                    trans[2] = float(translation[2])*scale
                    rotation = np.eye(3, 3)
                    tr = transforms3d.affines.compose(trans, rotation, [1,1,1])
                if chain[0][i].tag == 'rotation':
                    parameter = re.split(r'[ ]', chain[0][i].text)
                    angle = float(parameter[0])*math.pi/180
                    if parameter[1] == 'x':
                        tr = transforms3d.axangles.axangle2aff([1, 0, 0], angle, None)
                    if parameter[1] == 'y':
                        tr = transforms3d.axangles.axangle2aff([0, 1, 0], angle, None)
                    if parameter[1] == 'z':
                        tr = transforms3d.axangles.axangle2aff([0, 0, 1], angle, None)
                if chain[0][i].tag == 'rotationMatrix':
                    parameter = re.split(r'[ ]', chain[0][i].text)
                    rotation = np.zeros((3, 3))
                    rotation[0][0] = float(parameter[0])
                    rotation[1][0] = float(parameter[1])
                    rotation[2][0] = float(parameter[2])
                    rotation[0][1] = float(parameter[3])
                    rotation[1][1] = float(parameter[4])
                    rotation[2][1] = float(parameter[5])
                    rotation[0][2] = float(parameter[6])
                    rotation[1][2] = float(parameter[7])
                    rotation[2][2] = float(parameter[8])
                    trans = np.zeros(3)
                    tr = transforms3d.affines.compose(trans, rotation, [1,1,1])
                transform = np.matmul(transform, tr)
            # print(transform)
            #load joint
            joint_trans = []
            for joint in chain.iter('joint'):
                alg=re.findall(r'[+*-]', joint[0].text)
                dof_offset=re.findall(r"\d+\.?\d*", joint[0].text)
                if len(alg)<2:
                    id=int(dof_offset[0])
                    mul = 1.0
                    if len(alg)==0:
                        trans = 0
                    elif alg[0]=='+':
                        trans = float(dof_offset[1])*math.pi/180
                    elif alg[0]=='*': #0909
                        mul = mul * float(dof_offset[1])
                    else:
                        trans = -float(dof_offset[1])*math.pi/180
                if len(alg)==2:
                    id = int(dof_offset[0])
                    mul = float(dof_offset[1])
                    if alg[1]=='+':
                        trans = float(dof_offset[2])*math.pi/180
                    elif alg[1]=='*': #0909
                        mul = mul * float(dof_offset[1])
                    else:
                        trans = -float(dof_offset[2])*math.pi/180
                d = float(joint[1].text)*scale
                r = float(joint[2].text)*scale
                alpha = float(joint[3].text)*math.pi/180
                minV = float(joint[4].text)*math.pi/180
                maxV = float(joint[5].text)*math.pi/180
                joint_trans.append([id, mul, trans, d, r, alpha, minV, maxV])
            #load link
            i=0
            link_index=0
            parent=self.palm
            for link in chain.iter('link'):
                # print("Link:\t", link_index, link.text)
                xml_name = re.split(r'[.]', link.text)
                # print(xml_name[0], chain_index, link_index)
                if link.attrib['dynamicJointType'] == 'Universal':
                    parent=Link(self.linkMesh[xml_name[0]],xml_name[0],parent,transform,[joint_trans[i],joint_trans[i+1]])
                    parent.label = chain_index+1
                    parent.part_label = (parent.label * 10 + link_index)
                    # print(parent.part_label)
                    if self.contacts[chain_index,link_index]:
                        for j in range(len(self.contacts[chain_index,link_index])):
                            parent.add_end_effector(self.contacts[chain_index,link_index][j][0],self.contacts[chain_index,link_index][j][1])
                    i = i + 2 
                    link_index+=1
                else:
                    parent=Link(self.linkMesh[xml_name[0]],xml_name[0],parent,transform,[joint_trans[i]])
                    parent.label = chain_index+1
                    parent.part_label = (parent.label * 10 + link_index)
                    # print(parent.part_label)
                    if self.contacts[chain_index,link_index]:
                        for j in range(len(self.contacts[chain_index,link_index])):
                            parent.add_end_effector(self.contacts[chain_index,link_index][j][0],self.contacts[chain_index,link_index][j][1])
                    i = i + 1
                    link_index+=1
                transform = transforms3d.affines.compose(np.zeros(3), np.eye(3, 3), [1,1,1])
            for eef in chain.iter('end_effector'):
                location=[float(re.findall(r"\-*\d+\.?\d*", eef[0].text)[m])*scale for m in range(3)]
                normal=[float(re.findall(r"\-*\d+\.?\d*", eef[1].text)[m]) for m in range(3)]
                parent.add_end_effector(location,normal)
            chain_index+=1
        #eigen grasp
        self.read_eigen_grasp()
        if self.eg_num==0:
            self.use_eigen=False

    def get_labels(self): #0125
        self.labels = self.palm.get_labels()

    def get_part_labels(self):
        self.part_labels = self.palm.get_part_labels()

    def get_internal_labels(self):
        self.internal_labels = self.palm.get_internal_points()
        # print(np.sum(self.internal_labels))

    def read_eigen_grasp(self):
        if os.path.exists(self.hand_path+'/eigen'):
            for f in os.listdir(self.hand_path+'/eigen'):
                root=ET.parse(self.hand_path+'/eigen'+'/'+f).getroot()
                self.origin_eigen=np.zeros((self.nr_dof()),dtype=np.float64)
                self.dir_eigen=np.zeros((self.nr_dof(),2),dtype=np.float64)
                self.lb_eigen=np.zeros((2),dtype=np.float64)
                self.ub_eigen=np.zeros((2),dtype=np.float64)
                self.eg_num=0
                #read origin
                for ORIGIN in root.iter('ORIGIN'):
                    for DimVals in ORIGIN.iter('DimVals'):
                        for d in range(self.nr_dof()):
                            self.origin_eigen[d]=float(DimVals.attrib['d'+str(d)])
                #read directions
                off=0
                lb,ub=self.lb_ub()
                for EG in root.iter('EG'):
                    #initialize lb_eigen,ub_eigen
                    self.eg_num+=1
                    self.lb_eigen[off]=-np.finfo(np.float64).max
                    self.ub_eigen[off]= np.finfo(np.float64).max
                    #read non-zero entries
                    for DimVals in  EG.iter('DimVals'):
                        for d in range(self.nr_dof()):
                            if 'd'+str(d) in DimVals.attrib:
                                self.dir_eigen[d,off]=float(DimVals.attrib['d'+str(d)])
                                if self.dir_eigen[d,off]<0:
                                    tmp_lmt=(ub[d]-self.origin_eigen[d])/self.dir_eigen[d,off]
                                    self.lb_eigen[off]=max(self.lb_eigen[off],tmp_lmt)
                                    tmp_lmt=(lb[d]-self.origin_eigen[d])/self.dir_eigen[d,off]
                                    self.ub_eigen[off]=min(self.ub_eigen[off],tmp_lmt)
                                elif self.dir_eigen[d,off]>0:
                                    tmp_lmt=(ub[d]-self.origin_eigen[d])/self.dir_eigen[d,off]
                                    self.ub_eigen[off]=min(self.ub_eigen[off],tmp_lmt)
                                    tmp_lmt=(lb[d]-self.origin_eigen[d])/self.dir_eigen[d,off]
                                    self.lb_eigen[off]=max(self.lb_eigen[off],tmp_lmt)
                    off+=1
                assert off==2
                # read limits
                break

    def load_contacts(self,scale):
        contacts_file=self.hand_path+'contacts.xml'
        tree=ET.parse(contacts_file)
        root=tree.getroot()
        contacts=defaultdict(list)
        for virtual_contact in root.iter('virtual_contact'):
            finger_index=int(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",virtual_contact[0].text)[0])
            link_index=int(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",virtual_contact[1].text)[0])
            location=np.zeros(3)
            loc_string=re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", virtual_contact[4].text)
            location[0]=float(loc_string[0])*scale
            location[1]=float(loc_string[1])*scale
            location[2]=float(loc_string[2])*scale
            normal=np.zeros(3)
            nor_string=re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", virtual_contact[7].text)
            normal[0]=float(nor_string[0])
            normal[1]=float(nor_string[1])
            normal[2]=float(nor_string[2])
            contacts[finger_index,link_index].append([location,normal])
        return contacts

    def build_tensors(self):
        self.crossx=torch.Tensor([[ 0.0, 0.0, 0.0],
                                  [ 0.0, 0.0,-1.0],
                                  [ 0.0, 1.0, 0.0]]).view(3,3)
        self.crossy=torch.Tensor([[ 0.0, 0.0, 1.0],
                                  [ 0.0, 0.0, 0.0],
                                  [-1.0, 0.0, 0.0]]).view(3,3)
        self.crossz=torch.Tensor([[ 0.0,-1.0, 0.0],
                                  [ 1.0, 0.0, 0.0],
                                  [ 0.0, 0.0, 0.0]]).view(3,3)
                                  
    def nr_dof(self):
        return self.palm.max_dof_id()+1

    def lb_ub(self):
        lb=np.zeros(self.nr_dof())
        ub=np.zeros(self.nr_dof())
        self.palm.lb_ub(lb,ub)
        return lb,ub

    def forward_kinematics(self,extrinsic,dofs):
        ###
        if hasattr(self,'origin_eigen') and self.use_eigen:
            assert dofs.size==self.eg_num,('When using eigen, dof should be the same as eigen number')
            dofs=torch.from_numpy(np.asarray(dofs)).view(1,-1)
            if self.use_joint_limit:
                lb,ub=self.lb_eigen,self.ub_eigen
                lb=torch.from_numpy(lb).view(1,-1)
                ub=torch.from_numpy(ub).view(1,-1)
                sigmoid=torch.nn.Sigmoid()
                dofs=sigmoid(dofs)*(ub-lb)+lb
                dofs=torch.squeeze(dofs)
            dir_eigen=torch.from_numpy(self.dir_eigen).view([1,self.nr_dof(),self.eg_num])
            dofs=torch.matmul(dir_eigen,dofs.view(-1,2,1)).squeeze(2)
            dofs=dofs+torch.from_numpy(self.origin_eigen).view(1,self.nr_dof())
            dofs=torch.squeeze(dofs).numpy()
        else:
            if self.use_joint_limit:
                lb,ub=self.lb_ub()
                lb=torch.from_numpy(lb).view(1,-1)
                ub=torch.from_numpy(ub).view(1,-1)
                dofs=torch.from_numpy(np.asarray(dofs)).view(1,-1)
                sigmoid=torch.nn.Sigmoid()
                dofs=sigmoid(dofs)*(ub-lb)+lb
                dofs=torch.squeeze(dofs).numpy()
        ###
        if extrinsic.shape[0]==7:
            root_quat=np.asarray([float(extrinsic[3]),float(extrinsic[4]),float(extrinsic[5]),float(extrinsic[6])])
            # print(root_quat)
            root_rotation=transforms3d.quaternions.quat2mat(root_quat)
        else:
            assert extrinsic.shape[0]==6
            theta = np.linalg.norm(extrinsic[3:6])
            w = extrinsic[3:6]/max(theta,1e-6)
            K = np.array([[    0,-w[2], w[1]],
                          [ w[2],    0,-w[0]],
                          [-w[1], w[0],   0]])
            root_rotation = np.eye(3) + K*math.sin(theta) + np.matmul(K,K)*(1-math.cos(theta))
        root_translation = np.zeros(3)
        root_translation[0] = extrinsic[0]
        root_translation[1] = extrinsic[1]
        root_translation[2] = extrinsic[2]
        root_transform = transforms3d.affines.compose(root_translation, root_rotation, [1,1,1])
        self.palm.forward_kinematics(root_transform,dofs)

    def forward(self,params):
        #eigen grasp:
        if hasattr(self,'origin_eigen') and self.use_eigen:
            assert params.shape[1]==self.extrinsic_size+self.eg_num,\
                ('When using eigen, dof should be the same as eigen number')
            if self.use_quat:
                t,r,d=torch.split(params,[3,4,self.eg_num],dim=1)
            else:
                t,r,d=torch.split(params,[3,3,self.eg_num],dim=1)
            if self.use_joint_limit:
                lb,ub=self.lb_eigen,self.ub_eigen
                lb=torch.from_numpy(lb).view(1,-1).type(d.type())
                ub=torch.from_numpy(ub).view(1,-1).type(d.type())
                sigmoid=torch.nn.Sigmoid().type(d.type())
                d=sigmoid(d)*(ub-lb)+lb
            dir_eigen=torch.from_numpy(self.dir_eigen).view([1,self.nr_dof(),self.eg_num])
            d=torch.matmul(dir_eigen,d.view(-1,self.eg_num,1)).squeeze(2)
            d=d+torch.from_numpy(self.origin_eigen).view(1,self.nr_dof())
        else: 
            if self.use_quat:
                t,r,d=torch.split(params,[3,4,self.nr_dof()],dim=1)
            else:
                t,r,d=torch.split(params,[3,3,self.nr_dof()],dim=1)
            if self.use_joint_limit:
                lb,ub=self.lb_ub()
                lb=torch.from_numpy(lb).view(1,-1).type(d.type())
                ub=torch.from_numpy(ub).view(1,-1).type(d.type())
                sigmoid=torch.nn.Sigmoid().type(d.type())
                d=sigmoid(d)*(ub-lb)+lb
        #root rotation
        if self.use_quat:
            R=Quat2mat(r)
        else:
            theta=torch.norm(r,p=None,dim=1)
            theta=torch.clamp(theta,min=1e-6)
            w=r/theta.view([-1,1])
            wx,wy,wz=torch.split(w,[1,1,1],dim=1)
            K =wx.view([-1,1,1])*self.crossx.type(d.type())
            K+=wy.view([-1,1,1])*self.crossy.type(d.type())
            K+=wz.view([-1,1,1])*self.crossz.type(d.type())
            R=K*torch.sin(theta.view([-1,1,1]))+torch.matmul(K,K)*\
                (1-torch.cos(theta.view([-1,1,1])))+torch.eye(3).type(d.type())
        root_transform=torch.cat((R,t.view([-1,3,1])),dim=2)
        retp,retn,rett=self.palm.forward(root_transform,d)
        rett=torch.cat(rett,dim=2)
        return retp,retn,rett

    def forward_sampled(self,params): 
        #eigen grasp:
        if hasattr(self,'origin_eigen') and self.use_eigen:
            assert params.shape[1]==self.extrinsic_size+self.eg_num,\
                ('When using eigen, dof should be the same as eigen number')
            if self.use_quat:
                t,r,d=torch.split(params,[3,4,self.eg_num],dim=1)
            else:
                t,r,d=torch.split(params,[3,3,self.eg_num],dim=1)
            if self.use_joint_limit:
                lb,ub=self.lb_eigen,self.ub_eigen
                lb=torch.from_numpy(lb).view(1,-1).type(d.type())
                ub=torch.from_numpy(ub).view(1,-1).type(d.type())
                sigmoid=torch.nn.Sigmoid().type(d.type())
                d=sigmoid(d)*(ub-lb)+lb
            dir_eigen=torch.from_numpy(self.dir_eigen).view([1,self.nr_dof(),self.eg_num])
            d=torch.matmul(dir_eigen,d.view(-1,self.eg_num,1)).squeeze(2)
            d=d+torch.from_numpy(self.origin_eigen).view(1,self.nr_dof())
        else: 
            if self.use_quat:
                t,r,d=torch.split(params,[3,4,self.nr_dof()],dim=1)
            else:
                t,r,d=torch.split(params,[3,3,self.nr_dof()],dim=1)
            if self.use_joint_limit:
                lb,ub=self.lb_ub()
                lb=torch.from_numpy(lb).view(1,-1).type(d.type())
                ub=torch.from_numpy(ub).view(1,-1).type(d.type())
                sigmoid=torch.nn.Sigmoid().type(d.type())
                d=sigmoid(d)*(ub-lb)+lb
        #root rotation
        if self.use_quat:
            R=Quat2mat(r)
        else:
            theta=torch.norm(r,p=None,dim=1)
            theta=torch.clamp(theta,min=1e-6)
            w=r/theta.view([-1,1])
            wx,wy,wz=torch.split(w,[1,1,1],dim=1)
            K =wx.view([-1,1,1])*self.crossx.type(d.type())
            K+=wy.view([-1,1,1])*self.crossy.type(d.type())
            K+=wz.view([-1,1,1])*self.crossz.type(d.type())
            R=K*torch.sin(theta.view([-1,1,1]))+torch.matmul(K,K)*\
                (1-torch.cos(theta.view([-1,1,1])))+torch.eye(3).type(d.type())
        root_transform=torch.cat((R,t.view([-1,3,1])),dim=2)
        retp =self.palm.forward_sampled(root_transform,d)
        return retp


    def value_check(self,nr):
        if self.use_eigen:
            assert hasattr(self,'origin_eigen'),('Some hand does not apply to eigen')
            params=torch.randn(nr,self.extrinsic_size+self.eg_num)
        else:
            params=torch.randn(nr,self.extrinsic_size+self.nr_dof())
        pss,nss,_=self.forward(params)
        for i in range(pss.shape[0]):
            extrinsic=params.numpy()[i,0:self.extrinsic_size]
            dofs=params.numpy()[i,self.extrinsic_size:]
            self.forward_kinematics(extrinsic,dofs)
            pssi=[]
            nssi=[]
            for e in self.get_end_effector():
                pssi.append(e[0])
                nssi.append(e[1])
            pssi=torch.transpose(torch.tensor(pssi),0,1)
            nssi=torch.transpose(torch.tensor(nssi),0,1)
            pssi_diff=pssi.numpy()-pss.numpy()[i,]
            nssi_diff=nssi.numpy()-nss.numpy()[i,]
            print('pssNorm=%f pssErr=%f nssNorm=%f nssErr=%f'%
                  (np.linalg.norm(pssi),np.linalg.norm(pssi_diff),np.linalg.norm(nssi),np.linalg.norm(nssi_diff)))

    def grad_check(self,nr):
        if self.use_eigen:
            params=torch.randn(nr,self.extrinsic_size+self.eg_num)
        else:
            params=torch.randn(nr,self.extrinsic_size+self.nr_dof())
        params.requires_grad_()
        print('AutoGradCheck=',torch.autograd.gradcheck(self,(params),eps=1e-6,atol=1e-6,rtol=1e-6,raise_exception=True))

    def draw(self,scale_factor=1,show_to_screen=True):
        mesh=self.palm.draw()
        mesh.apply_scale(scale_factor)
        if show_to_screen:
            mesh.show()
        return mesh

    def write_limits(self):
        if os.path.exists('limits'):
            shutil.rmtree('limits')
        os.mkdir('limits')
        lb,ub=hand.lb_ub()
        
        for i in range(len(lb)):
            dofs=np.asarray([0.0 for i in range(hand.nr_dof())])
            self.use_eigen=False
            dofs[i]=lb[i]
            hand.forward_kinematics(np.zeros(self.extrinsic_size),dofs)
            mesh_vtk=trimesh_to_vtk(self.draw(1,False))
            write_vtk(mesh_vtk,'limits/lower%d.vtk'%i)
            
            dofs[i]=ub[i]
            hand.forward_kinematics(np.zeros(self.extrinsic_size),dofs)
            mesh_vtk=trimesh_to_vtk(self.draw(1,False))
            write_vtk(mesh_vtk,'limits/upper%d.vtk'%i)
            
        if hasattr(self,'origin_eigen'):
            for d in range(self.lb_eigen.shape[0]):
                dofs=self.origin_eigen+self.dir_eigen[:,d]*self.lb_eigen[d]
                hand.forward_kinematics(np.zeros(self.extrinsic_size),dofs)
                mesh_vtk=trimesh_to_vtk(self.draw(1,False))
                write_vtk(mesh_vtk,'limits/lowerEigen%d.vtk'%d)
                
                dofs=self.origin_eigen+self.dir_eigen[:,d]*self.ub_eigen[d]
                hand.forward_kinematics(np.zeros(self.extrinsic_size),dofs)
                mesh_vtk=trimesh_to_vtk(self.draw(1,False))
                write_vtk(mesh_vtk,'limits/upperEigen%d.vtk'%d)

    def random_dofs(self):
        lb,ub=self.lb_ub()
        dofs=np.zeros(self.nr_dof())
        for i in range(len(dofs)):
            dofs[i]=random.uniform(lb[i],ub[i])
        return dofs
    
    def random_lmt_dofs(self):
        lb,ub=self.lb_ub()
        dofs=np.zeros(self.nr_dof())
        for i in range(len(dofs)):
            dofs[i]=lb[i] if random.uniform(0.0,1.0)<0.5 else ub[i]
        return dofs

    def get_end_effector(self):
        return self.palm.get_end_effector_all()

    def energy_map_back_link(self,link,hulls,cones):
        obj=0
        for ee in link.get_end_effector():
            #location
            loc=np.matmul(link.joint_transform[0:3,0:3],ee[0])
            loc=np.add(link.joint_transform[0:3,3],loc)
            loc=np.subtract(loc,np.asarray(hulls[self.ticks][0]))
            obj+=loc.dot(loc)
            #normal
            dir=np.matmul(link.joint_transform[0:3,0:3],ee[1])
            dir=np.subtract(dir,np.asarray(cones[self.ticks][0]))
            obj+=dir.dot(dir)
            self.ticks+=1
        for c in link.children:
            obj+=self.energy_map_back_link(c,hulls,cones)
        return obj

    def energy_map_back(self,x0,x,hulls,cones,reg):
        dx=np.subtract(x0,x)
        self.ticks=0
        self.forward_kinematics(x[0:7],x[7:])
        obj=self.energy_map_back_link(self.palm,hulls,cones)+dx.dot(dx)*reg
        delattr(self,'ticks')
        return obj

    def map_back(self,extrinsic,dofs,hulls,cones,reg=1e-6):
        lb,ub=self.lb_ub()
        bnds=[(None,None) for i in range(7)]
        for i in range(len(lb)):
            bnds.append((lb[i],ub[i]))
        x0=np.concatenate((extrinsic,dofs))
        fun=lambda x: self.energy_map_back(x0,x,hulls,cones,reg)
        res=scipy.optimize.minimize(fun,x0,method='SLSQP',bounds=tuple(bnds))
        return res.x[0:7],res.x[7:]

    def sample(self,count,re_sample=False):
        # count is the total sampled points
        total_area=self.palm.add_area()
        count_over_area=count/total_area
        self.palm.sample(count_over_area,self.hand_path+'off/',re_sample)
        self.get_labels()
        self.get_part_labels()
        self.get_internal_labels()

    def sample_fwk(self):
        sample_points=self.palm.sample_fwk()
        return sample_points

    def sample_fwk_normal(self):
        sample_normals = self.palm.sample_fwk_normal()
        return sample_normals
        

def vtk_add_from_hand(hand,renderer,scale):
    #palm and fingers 
    mesh=hand.draw(scale_factor=1,show_to_screen=False)
    vtk_mesh=trimesh_to_vtk(mesh)
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(vtk_mesh)
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().SetOpacity(1)
    renderer.AddActor(mesh_actor)

    #end effectors
    end_effector=hand.get_end_effector()
    for i in range(len(end_effector)):
        #point
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(end_effector[i][0][0],end_effector[i][0][1],end_effector[i][0][2])
        sphere.SetRadius(3*scale)
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(24)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(.0/255,.0/255,255.0/255)

        #normal
        normal=vtk.vtkArrowSource()
        normal.SetTipResolution(100)
        normal.SetShaftResolution(100)
        # Generate a random start and end point
        startPoint=[end_effector[i][0][0],end_effector[i][0][1],end_effector[i][0][2]]
        endPoint=[0]*3
        rng = vtk.vtkMinimalStandardRandomSequence()
        rng.SetSeed(8775070)  # For testing.
        n=[end_effector[i][1][0],end_effector[i][1][1],end_effector[i][1][2]]
        direction=[None,None,None]
        direction[0]=n[0]
        direction[1]=n[1]
        direction[2]=n[2]
        for j in range(0, 3):
            endPoint[j]=startPoint[j]+direction[j]*20*scale
        # Compute a basis
        normalizedX = [0 for i in range(3)]
        normalizedY = [0 for i in range(3)]
        normalizedZ = [0 for i in range(3)]
        # The X axis is a vector from start to end
        vtk.vtkMath.Subtract(endPoint, startPoint, normalizedX)
        length = vtk.vtkMath.Norm(normalizedX)
        vtk.vtkMath.Normalize(normalizedX)
        # The Z axis is an arbitrary vector cross X
        arbitrary=[0 for i in range(3)]
        for j in range(0, 3):
            rng.Next()
            arbitrary[j] = rng.GetRangeValue(-10, 10)
        vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
        vtk.vtkMath.Normalize(normalizedZ)
        # The Y axis is Z cross X
        vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtk.vtkMatrix4x4()
        # Create the direction cosine matrix
        matrix.Identity()
        for j in range(0, 3):
            matrix.SetElement(j, 0, normalizedX[j])
            matrix.SetElement(j, 1, normalizedY[j])
            matrix.SetElement(j, 2, normalizedZ[j])
        # Apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(startPoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)
        # Transform the polydata
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(normal.GetOutputPort())
        # Create a mapper and actor for the arrow
        normalMapper = vtk.vtkPolyDataMapper()
        normalActor = vtk.vtkActor()
        USER_MATRIX = True
        if USER_MATRIX:
            normalMapper.SetInputConnection(normal.GetOutputPort())
            normalActor.SetUserMatrix(transform.GetMatrix())
        else:
            normalMapper.SetInputConnection(transformPD.GetOutputPort())
        normalActor.SetMapper(normalMapper)
        normalActor.GetProperty().SetColor(255.0/255, 0.0/255, 0.0/255)

        renderer.AddActor(normalActor)
        renderer.AddActor(sphere_actor)
    
def vtk_render(renderer,axes=True):
    if axes is True:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(5, 5, 5)
        axes.SetTipTypeToCone()
        axes.SetConeRadius(0.25)
        axes.SetConeResolution(50)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.01)
        axes.SetNormalizedLabelPosition(1,1,1)
        renderer.AddActor(axes)
    renderer.SetBackground(0.329412, 0.34902, 0.427451)
    renderer.ResetCamera()
    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(1000, 1000)
    renderWindow.AddRenderer(renderer)
    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    # Begin Interaction
    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__=='__main__':
    # hand_paths=['hand/ShadowHand/','hand/BarrettHand/']
    hand_paths=['hand/ShadowHand/']
    # hand_paths=['hand/BarrettHand/']
    scale=0.01
    for path in hand_paths:
        use_eigen=False
        hand=Hand(path,scale,use_joint_limit=False,use_quat=True,use_eigen=use_eigen)
        hand.sample(13000, re_sample=False)
        if hand.use_eigen:
            dofs=np.zeros(hand.eg_num)
        else:
            dofs=np.zeros(hand.nr_dof())
        dofs[-5:] = [-0.3, 1.22, 1, 1, 1]
        edofs = np.zeros(hand.extrinsic_size)
        hand.forward_kinematics(edofs,dofs)
        # hand.draw()
        start = time.time()
        sample_points0=hand.sample_fwk() # Sampled Point FWKz
        sample_normal0=hand.sample_fwk_normal()

        selected = hand.internal_labels > -10
        sample_points = sample_points0[selected]
        sample_normal = sample_normal0[selected]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample_points)
        # pcd.normals = o3d.utility.Vector3dVector(sample_normal)
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        end = time.time()
        print("Time: ", end-start)

        # Visualization (VTK)
        # points_vtk=points_to_vtk(sample_points)
        # hand_vtk=trimesh_to_vtk(hand.draw(show_to_screen=False))
        # write_vtk(points_vtk,'points.vtk')
        # write_vtk(hand_vtk,'hand.vtk')
        # hand.value_check(10)
        # hand.grad_check(2)
        # hand.write_limits()
        # renderer=vtk.vtkRenderer()
        # vtk_add_from_hand(hand,renderer,scale)
        # vtk_render(renderer,axes=False)
        # renderer=vtk.vtkRenderer()
        # vtk_add_from_hand(hand,renderer,scale)
        # vtk_render(renderer,axes=False)

