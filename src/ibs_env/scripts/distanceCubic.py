import torch,os,random,vtk
import numpy as np
from hand import Hand
from distance import Distance,cal_norm,cal_norm_batch,filter_files,SDF,test_SDF
torch.set_default_dtype(torch.float64)

def interp1d_cubic(id,frac,SDFGrid,grad=False): 
    #id is a list of size 4
    #frac is a list of size 3: [fracX,fracY,fracZ]
    id0=id[0].long()
    id1=id[1].long()
    id2=id[2].long()
    id3=id[3].long()
    vkm1=SDFGrid.tensor[id0[0]][id0[1]][id0[2]]
    vk=SDFGrid.tensor[id1[0]][id1[1]][id1[2]]
    vkp1=SDFGrid.tensor[id2[0]][id2[1]][id2[2]]
    vkp2=SDFGrid.tensor[id3[0]][id3[1]][id3[2]]
    if grad:
        return (3*frac[2]**2*(3*vk-vkm1-3*vkp1+vkp2))/2+frac[2]*(-5*vk+2*vkm1+4*vkp1-vkp2)/2+(vkp1-vkm1)/2
    else:
        a=(3*vk-vkm1-3*vkp1+vkp2)/2
        b=-(5*vk-2*vkm1-4*vkp1+vkp2)/2
        c=(vkp1-vkm1)/2
        d=vk
        return a*frac[2]**3+b*frac[2]**2+c*frac[2]+d
def interp2d_cubic(id,frac,SDFGrid,grad=False): 
    #id is a list of size 4x4
    #frac is a list of size 3: [fracX,fracY,fracZ]
    vkm1=interp1d_cubic(id[0],frac,SDFGrid,grad)
    vk=interp1d_cubic(id[1],frac,SDFGrid,grad)
    vkp1=interp1d_cubic(id[2],frac,SDFGrid,grad)
    vkp2=interp1d_cubic(id[3],frac,SDFGrid,grad)
    a=(3*vk-vkm1-3*vkp1+vkp2)/2
    b=-(5*vk-2*vkm1-4*vkp1+vkp2)/2
    c=(vkp1-vkm1)/2
    d=vk
    return a*frac[1]**3+b*frac[1]**2+c*frac[1]+d
def interp3d_cubic(id,frac,SDFGrid,grad=False): 
    #id is a list of size 4x4x4
    #frac is a list of size 3: [fracX,fracY,fracZ]
    vkm1=interp2d_cubic(id[0],frac,SDFGrid,grad)
    vk=interp2d_cubic(id[1],frac,SDFGrid,grad)
    vkp1=interp2d_cubic(id[2],frac,SDFGrid,grad)
    vkp2=interp2d_cubic(id[3],frac,SDFGrid,grad)
    a=(3*vk-vkm1-3*vkp1+vkp2)/2
    b=-(5*vk-2*vkm1-4*vkp1+vkp2)/2
    c=(vkp1-vkm1)/2
    d=vk
    return a*frac[0]**3+b*frac[0]**2+c*frac[0]+d

def interp1d_cubic_batch(id,frac,SDFGrid,grad=False): 
    #id is a list of size 4
    #frac is a list of size 3: [fracX,fracY,fracZ]
    id0=id[0].long()
    id1=id[1].long()
    id2=id[2].long()
    id3=id[3].long()
    for i in range(len(SDFGrid)):
        SDF0i=SDFGrid[i].tensor[id0[i,:,0],id0[i,:,1],id0[i,:,2]].unsqueeze(0)
        SDF1i=SDFGrid[i].tensor[id1[i,:,0],id1[i,:,1],id1[i,:,2]].unsqueeze(0)
        SDF2i=SDFGrid[i].tensor[id2[i,:,0],id2[i,:,1],id2[i,:,2]].unsqueeze(0)
        SDF3i=SDFGrid[i].tensor[id3[i,:,0],id3[i,:,1],id3[i,:,2]].unsqueeze(0)
        vkm1=SDF0i if i==0 else torch.cat([vkm1,SDF0i],dim=0)
        vk=SDF1i if i==0 else torch.cat([vk,SDF1i],dim=0)
        vkp1=SDF2i if i==0 else torch.cat([vkp1,SDF2i],dim=0)
        vkp2=SDF3i if i==0 else torch.cat([vkp2,SDF3i],dim=0)
    if grad:
        return (3*frac[:,:,2]**2*(3*vk-vkm1-3*vkp1+vkp2))/2+frac[:,:,2]*(-5*vk+2*vkm1+4*vkp1-vkp2)/2+(vkp1-vkm1)/2
    else:
        a=(3*vk-vkm1-3*vkp1+vkp2)/2
        b=-(5*vk-2*vkm1-4*vkp1+vkp2)/2
        c=(vkp1-vkm1)/2
        d=vk
        return a*frac[:,:,2]**3+b*frac[:,:,2]**2+c*frac[:,:,2]+d
def interp2d_cubic_batch(id,frac,SDFGrid,grad=False): 
    #id is a list of size 4x4
    #frac is a list of size 3: [fracX,fracY,fracZ]
    vkm1=interp1d_cubic_batch(id[0],frac,SDFGrid,grad)
    vk=interp1d_cubic_batch(id[1],frac,SDFGrid,grad)
    vkp1=interp1d_cubic_batch(id[2],frac,SDFGrid,grad)
    vkp2=interp1d_cubic_batch(id[3],frac,SDFGrid,grad)
    a=(3*vk-vkm1-3*vkp1+vkp2)/2
    b=-(5*vk-2*vkm1-4*vkp1+vkp2)/2
    c=(vkp1-vkm1)/2
    d=vk
    return a*frac[:,:,1]**3+b*frac[:,:,1]**2+c*frac[:,:,1]+d
def interp3d_cubic_batch(id,frac,SDFGrid,grad=False): 
    #id is a list of size 4x4x4
    #frac is a list of size 3: [fracX,fracY,fracZ]
    vkm1=interp2d_cubic_batch(id[0],frac,SDFGrid,grad)
    vk=interp2d_cubic_batch(id[1],frac,SDFGrid,grad)
    vkp1=interp2d_cubic_batch(id[2],frac,SDFGrid,grad)
    vkp2=interp2d_cubic_batch(id[3],frac,SDFGrid,grad)
    a=(3*vk-vkm1-3*vkp1+vkp2)/2
    b=-(5*vk-2*vkm1-4*vkp1+vkp2)/2
    c=(vkp1-vkm1)/2
    d=vk
    return a*frac[:,:,0]**3+b*frac[:,:,0]**2+c*frac[:,:,0]+d

class DistanceCubic(Distance):
    def forward(self,pss,sdfs):
        pss=pss.transpose(1,2)
        assert(pss.shape[0]==len(sdfs))
        for i in range(len(sdfs)):
            firsti=sdfs[i].first.view([1,1,3])
            coefi=sdfs[i].coef.view([1,1,3])
            maxli=sdfs[i].max_limit.view([1,1,3])
            first=firsti if i==0 else torch.cat([first,firsti],dim=0)
            coef=coefi if i==0 else torch.cat([coef,coefi],dim=0)
            maxl=maxli if i==0 else torch.cat([maxl,maxli],dim=0)
        id=[[[None for i in range(4)]for j in range(4)]for k in range(4)]
        frac=torch.clamp(torch.min((pss-first)*coef,maxl.double()-1.0),min=1.0)
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    id[ii][jj][kk]=torch.stack((torch.floor(frac[:,:,0])+ii-1,torch.floor(frac[:,:,1])+jj-1,torch.floor(frac[:,:,2])+kk-1),2)
        frac=frac-id[1][1][1]
        dss=interp3d_cubic_batch(id,frac,sdfs)
        nss=cal_norm_batch(id,frac,sdfs,interp3d_cubic_batch)
        return dss,nss,torch.clamp(dss,max=0.0).sum()*(-1)
    
    def forward_validation(self,pss,sdfs):
        pss=pss.transpose(1,2)
        batch_size=pss.shape[0]
        assert batch_size == len(sdfs)
        dss=np.zeros((batch_size,pss.shape[1]))
        nss=np.zeros((batch_size,pss.shape[1],3))
        loss=0
        id=[[[None for i in range(4)]for j in range(4)]for k in range(4)]
        for i in range(batch_size):
            for j in range(pss.shape[1]):
                frac=torch.zeros(3)
                for d in range(3):
                    frac[d]=(pss[i,j,d]-sdfs[i].first[d])*sdfs[i].coef[d]
                    frac[d]=torch.clamp(frac[d],min=1.0,max=sdfs[i].max_limit[d]-1.0)
                for ii in range(4):
                    for jj in range(4):
                        for kk in range(4):
                            id[ii][jj][kk]=torch.tensor([torch.floor(frac[0])+ii-1,torch.floor(frac[1])+jj-1,torch.floor(frac[2])+kk-1])
                frac=frac-id[1][1][1]
                dss[i,j]=interp3d_cubic(id,frac,sdfs[i])
                nss[i,j,:]=cal_norm(id,frac,sdfs[i],interp3d_cubic)
                loss+=min(dss[i,j],0.0)
        return dss,nss,loss*(-1)

if __name__=='__main__':
    #test sphere
    #sdf=SDF('sphere.vtk',rad=0.5)
    #test_SDF(sdf,type=DistanceCubic)
    
    sdf_list=[]
    filter_files('sdf','vtk',sdf_list)
    sdfs=[SDF(f) for f in sdf_list]
    if len(sdfs)==0:
        sdfs.append(SDF(None,rad=200.0))
    
    hand_paths=['hand/BarrettHand/','hand/ShadowHand/']
    for path in hand_paths:
        hand=Hand(path,1,True)
        my_distance=DistanceCubic()
        my_distance.value_check(hand,sdfs)
        my_distance.grad_check(hand,sdfs)