#!/usr/bin/env python
'''
    This file is from http:\\
'''

import torch, os, trimesh
import numpy as np
import ctypes as ct
from hand import Hand
from distance import filter_files
from distanceExact import find_lib_path

# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)

import time
import vtk
from hand import vtk_add_from_hand, vtk_render
from torch.optim import Adam,SGD

class SelfCollision(torch.autograd.Function):
    mesh_paths=None
    deepest=True
    #load ctype
    clib=ct.cdll.LoadLibrary(find_lib_path())
    clib.writeSelfColl.argtypes=[ct.c_char_p]
    clib.readSelfColl.argtypes=[ct.c_char_p]
    clib.createSelfColl.argtypes=[]
    clib.printSelfColl.argtypes=[]
    clib.assemble.argtypes=[]
    clib.prune.argtypes=[ct.c_int,ct.c_int]
    clib.addLink.argtypes=[ct.POINTER(ct.c_double),ct.c_int,ct.c_double]
    clib.detect.argtypes=[ct.POINTER(ct.c_double),ct.POINTER(ct.c_char_p),ct.c_int,ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_bool]
    clib.writeVTK.argtypes=[ct.POINTER(ct.c_double),ct.POINTER(ct.c_char_p),ct.c_int,ct.c_char_p]
        
    @staticmethod
    def initLink(link,rad):
        #assign id
        link.self_coll_id=len(SelfCollision.links)
        SelfCollision.links.append(link)
        #add link
        vss=np.array(link.mesh.vertices)
        vssPtr=vss.ctypes.data_as(ct.POINTER(ct.c_double))
        assert link.self_coll_id==SelfCollision.clib.addLink(vssPtr,len(link.mesh.vertices),rad)
        #add children links
        for c in link.children:
            SelfCollision.initLink(c,rad)
        #add prune
        for c in link.children:
            SelfCollision.clib.prune(link.self_coll_id,c.self_coll_id)
            for cc in c.children:
                SelfCollision.clib.prune(link.self_coll_id,cc.self_coll_id)
        
    @staticmethod
    def init(hand,rad,force=False):
        SelfCollision.hand=hand
        SelfCollision.debugWrite=False
        path=hand.hand_path+"/SelfColl"+str(rad)+".dat"
        if not force and os.path.exists(path):
            SelfCollision.clib.readSelfColl(path.encode())
        else:
            SelfCollision.links=[]
            SelfCollision.clib.createSelfColl()
            SelfCollision.initLink(hand.palm,rad)
            SelfCollision.clib.assemble()
            SelfCollision.clib.writeSelfColl(path.encode())
        SelfCollision.clib.printSelfColl()

    @staticmethod
    def buffet(dofs=25):
        if dofs == 25:
            hand = Hand('src/ibs_env/scripts/hand/ShadowHand/', 0.01, use_joint_limit=False, use_quat=True, use_eigen=False)
        else:
            hand = Hand('src/ibs_env/scripts/hand/ShadowHand/', 0.01, use_joint_limit=False, use_quat=False, use_eigen=False)
        SelfCollision.init(hand,0.01)
        
    @staticmethod
    def forward(ctx, tss):
        if SelfCollision.mesh_paths is None:
            names=ct.POINTER(ct.c_char_p)()
        else:
            assert len(SelfCollision.mesh_paths)==tss.shape[0]
            names=(ct.c_char_p*tss.shape[0])()
            for i in range(len(SelfCollision.mesh_paths)):
                names[i]=SelfCollision.mesh_paths[i].encode()
        #tss Data
        tssD=tss.reshape(-1).detach().cpu().numpy().astype(np.float64)
        tssDPtr=tssD.ctypes.data_as(ct.POINTER(ct.c_double))
        #dtss Data
        dtssD=np.zeros(tssD.shape,dtype=np.float64)
        dtssDPtr=dtssD.ctypes.data_as(ct.POINTER(ct.c_double))
        #loss Data
        lossD=np.zeros(tss.shape[0],dtype=np.float64)
        lossDPtr=lossD.ctypes.data_as(ct.POINTER(ct.c_double))
        #call C function
        SelfCollision.clib.detect(tssDPtr,names,tss.shape[0],dtssDPtr,lossDPtr,SelfCollision.deepest)
        if SelfCollision.debugWrite:
            SelfCollision.clib.writeVTK(tssDPtr,names,tss.shape[0],'debugWriteVTK'.encode())
        dtssD=torch.Tensor(-dtssD).reshape(-1,tss.shape[1],tss.shape[2])
        lossD=torch.Tensor(-lossD)
        ctx.save_for_backward(dtssD,lossD)
        return lossD.type(tss.type())
        
    @staticmethod
    def backward(ctx, lossG):
        dtssD,lossD=ctx.saved_tensors
        return lossG.view([-1,1,1])*dtssD.type(lossG.type())
    
    @staticmethod
    def value_grad_check(n=3):
        pss=torch.randn(n,SelfCollision.hand.nr_dof()+7)
        sss = time.time()
        _,_,tss=SelfCollision.hand.forward(pss)
        # value check
        SelfCollision.debugWrite=True
        SelfCollision.apply(tss)
        # gradient check
        SelfCollision.debugWrite=False
        eee = time.time()
        print(eee-sss)
        tssTest=torch.Tensor(tss.cpu().numpy())
        tssTest.requires_grad_()
        # print('AutoGradCheck=',torch.autograd.gradcheck(SelfCollision.apply,(tssTest),eps=1e-6,atol=1e-6,rtol=1e-5,raise_exception=False))
    
    @staticmethod
    def calculate(pss):
        _,_,tss=SelfCollision.hand.forward(pss)
        value = SelfCollision.apply(tss)
        return value


    @staticmethod
    def test_grad():
        pss=torch.randn(1,SelfCollision.hand.nr_dof()+7)
        pss.requires_grad_()
        optim = SGD([pss], lr=1e-2)
        counter = 0
        last_dofs = pss.detach().numpy()
        while True:
            off = torch.zeros(25)
            off[:7] = 100
            value = SelfCollision.calculate(pss+off)
            print(value)
            optim.zero_grad()
            value.sum().backward()
            # print(pss.grad)
            # print(pss)
            optim.step()
            if counter % 100 == 0:
                renderer=vtk.vtkRenderer()
                dofs = pss.detach().numpy()
                SelfCollision.hand.forward_kinematics(dofs[0,:7], dofs[0,7:])
                vtk_add_from_hand(SelfCollision.hand,renderer,0.01)
                vtk_render(renderer,axes=False)
                last_dofs = dofs
            counter += 1
            # print(value.sum())


if __name__=='__main__':
    # hand_paths=['hand/ShadowHand/','hand/BarrettHand/']
    hand_paths=['hand/ShadowHand/']
    mesh_paths=[]
    # data_list = filter_files('sdf','PSet.dat',mesh_paths,abs=True)
    # SelfCollision.mesh_paths=mesh_paths
    scale=0.01
    for path in hand_paths:
        # hand=Hand(path,scale,True)
        # hand = Hand('hand/ShadowHand/', 0.01, use_joint_limit=False, use_quat=True, use_eigen=False)
        # SelfCollision.init(hand,0.01)
        SelfCollision.buffet()
        # SelfCollision.value_grad_check(n=16)
        SelfCollision.test_grad()
