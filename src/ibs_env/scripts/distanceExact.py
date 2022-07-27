import torch,os,random,vtk,pickle
from distance import filter_files,SDF
from hand import Hand
import ctypes as ct
import numpy as np
from distanceCubic import DistanceCubic
torch.set_default_dtype(torch.float64)

def find_lib_path():
    # path=os.path.abspath('../')
    path=os.path.abspath('./')
    for p in os.listdir(path):
        full_path=path+'/'+p
        if os.path.exists(full_path+'/libPythonInterface.so'):
            return full_path+'/libPythonInterface.so'
        if os.path.exists(full_path+'/libPythonInterface.a'):
            return full_path+'/libPythonInterface.a'
    assert False
    
class DistanceExact(torch.autograd.Function):
    mesh_paths=None
    #load ctype
    clib=ct.cdll.LoadLibrary(find_lib_path())
    #void init();
    clib.distance.argtypes=[ct.POINTER(ct.c_char_p),ct.c_int,   \
                            ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_int]
        
    @staticmethod
    def forward(ctx, pss):
        names=(ct.c_char_p*pss.shape[0])()
        for i in range(len(DistanceExact.mesh_paths)):
            names[i]=DistanceExact.mesh_paths[i].encode()
        #pss Data
        pssD=pss.reshape(-1).detach().cpu().numpy().astype(np.float64)
        pssDPtr=pssD.ctypes.data_as(ct.POINTER(ct.c_double))
        #dss Data
        dssD=np.zeros((pssD.shape[0]//3),dtype=np.float64)
        dssDPtr=dssD.ctypes.data_as(ct.POINTER(ct.c_double))
        #nss Data
        nssD=np.zeros(pssD.shape[0],dtype=np.float64)
        nssDPtr=nssD.ctypes.data_as(ct.POINTER(ct.c_double))
        #hss Data
        hssD=np.zeros((pssD.shape[0]*3),dtype=np.float64)
        hssDPtr=hssD.ctypes.data_as(ct.POINTER(ct.c_double))
        #call C function
        DistanceExact.clib.distance(names,len(names),pssDPtr,dssDPtr,nssDPtr,hssDPtr,pss.shape[2])
        #loss
        dssD=torch.reshape(torch.tensor(dssD),(-1,pss.shape[2]))
        nssD=torch.reshape(torch.tensor(nssD),(-1,pss.shape[2],3))
        hssD=torch.reshape(torch.tensor(hssD),(-1,pss.shape[2],3,3))
        loss=torch.clamp(dssD,max=0.0).sum()*(-1)
        ctx.save_for_backward(dssD,nssD,hssD,loss)
        return dssD.type(pss.type()),nssD.type(pss.type()),loss.type(pss.type())
        
    @staticmethod
    def backward(ctx, dssDG, nssDG, lossG):
        dssD,nssD,hssD,loss=ctx.saved_tensors
        #dssDG
        pssG=nssD.type(dssDG.type())*torch.unsqueeze(dssDG,dim=2)
        #nssDG
        pssG+=torch.squeeze(torch.matmul(hssD.type(nssDG.type()),torch.unsqueeze(nssDG,dim=3)),dim=3)
        #lossG
        dssDSgn=torch.where(dssD<0,torch.tensor([-1.]),torch.tensor([0.]))
        dssDSgn=torch.unsqueeze(dssDSgn,dim=2)
        pssG+=(nssD*dssDSgn*lossG).type(pssG.type())
        return pssG.transpose(1,2)
        
    def grad_check(hand,mesh_paths):
        configurations=torch.randn(len(mesh_paths),hand.nr_dof()+6)
        pss,h_nss,_=hand.forward(configurations)
        DistanceExact.mesh_paths=mesh_paths
        pssTest=torch.autograd.Variable(torch.randn(pss.shape),requires_grad=True)
        print('AutoGradCheck=',torch.autograd.gradcheck(DistanceExact.apply,(pssTest),eps=1e-6,atol=1e-6,rtol=1e-5,raise_exception=True))

    def value_check(hand,mesh_paths):
        N=1
        mesh_path=None
        for i in range(len(mesh_paths)):
            if 'sphere_unit' in mesh_paths[i]:
                mesh_path=[mesh_paths[i]]*N
                break
        if mesh_path is None:
            print("Can not find test unit sphere!")
            exit()
        pss=torch.DoubleTensor(N,3,2)
        #instance
        for i in range(N):
            pss[i,(i+0)%3,0]=0
            pss[i,(i+1)%3,0]=(i+1)/(N+1)*0.5
            pss[i,(i+2)%3,0]=0
            pss[i,(i+0)%3,1]=0
            pss[i,(i+1)%3,1]=0
            pss[i,(i+2)%3,1]=(i+1)/(N+1)*0.5
        sdfs=[SDF(None,rad=0.5)]*N
        dc=DistanceCubic()
        dssRef,nssRef,lossRef=dc.forward(pss,sdfs)
        DistanceExact.mesh_paths=mesh_path
        dss,nss,loss=DistanceExact.apply(pss)
        print('dssErr=',np.linalg.norm(dss.numpy()-dssRef.numpy()))
        print('nssErr=',np.linalg.norm(nss.numpy()-nssRef.numpy()))
        print('lossErr=',loss-lossRef)

if __name__=='__main__':
    mesh_paths=[]
    filter_files('sdf','BVH.dat',mesh_paths,abs=True)
    hand_paths=['hand/BarrettHand/','hand/ShadowHand/']
    for path in hand_paths:
        hand=Hand(path,1,True)
        DistanceExact.value_check(hand,mesh_paths)
        DistanceExact.grad_check(hand,mesh_paths)