import torch,os,random,vtk,pickle
from hand import Hand
import numpy as np
torch.set_default_dtype(torch.float64)

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

def interp1d(id,frac,SDFGrid,grad=False): 
    #id is a list of size 2
    #frac is a list of size 3: [fracX,fracY,fracZ]
    id0=id[0].long()
    id1=id[1].long()
    if grad:
        return SDFGrid.tensor[id1[0]][id1[1]][id1[2]]-SDFGrid.tensor[id0[0]][id0[1]][id0[2]]
    else: return SDFGrid.tensor[id0[0]][id0[1]][id0[2]]*(1-frac[2])+SDFGrid.tensor[id1[0]][id1[1]][id1[2]]*frac[2]
def interp2d(id,frac,SDFGrid,grad=False): 
    #id is a list of size 2x2
    #frac is a list of size 3: [fracX,fracY,fracZ]
    return interp1d(id[0],frac,SDFGrid,grad)*(1-frac[1])+interp1d(id[1],frac,SDFGrid,grad)*frac[1]
def interp3d(id,frac,SDFGrid,grad=False): 
    #id is a list of size 2x2x2
    #frac is a list of size 3: [fracX,fracY,fracZ]
    return interp2d(id[0],frac,SDFGrid,grad)*(1-frac[0])+interp2d(id[1],frac,SDFGrid,grad)*frac[0]

def interp1d_batch(id,frac,SDFGrid,grad=False):
    id0=id[0].long()
    id1=id[1].long()
    for i in range(len(SDFGrid)):
        SDF0i=SDFGrid[i].tensor[id0[i,:,0],id0[i,:,1],id0[i,:,2]].unsqueeze(0)
        SDF1i=SDFGrid[i].tensor[id1[i,:,0],id1[i,:,1],id1[i,:,2]].unsqueeze(0)
        SDF0=SDF0i if i==0 else torch.cat([SDF0,SDF0i],dim=0)
        SDF1=SDF1i if i==0 else torch.cat([SDF1,SDF1i],dim=0)
    if grad:
        return SDF1-SDF0
    else: return SDF0*(1-frac[:,:,2])+SDF1*frac[:,:,2]
def interp2d_batch(id,frac,SDFGrid,grad=False): 
    return interp1d_batch(id[0],frac,SDFGrid,grad)*(1-frac[:,:,1])+interp1d_batch(id[1],frac,SDFGrid,grad)*frac[:,:,1]
def interp3d_batch(id,frac,SDFGrid,grad=False):
    return interp2d_batch(id[0],frac,SDFGrid,grad)*(1-frac[:,:,0])+interp2d_batch(id[1],frac,SDFGrid,grad)*frac[:,:,0]

def swap_index(id,frac):
    nrX=len(id)
    idr=[[[None for k in range(nrX)] for j in range(nrX)] for i in range(nrX)]
    fracr=[frac[2],frac[0],frac[1]]
    for i in range(nrX):
        for j in range(nrX):
            for k in range(nrX):
                idr[k][i][j]=id[i][j][k]
    return idr,fracr
def cal_norm(id,frac,SDFGrid,interp_fn,eps=1e-5):
    normal=torch.zeros(3)
    normal[2]=interp_fn(id,frac,SDFGrid,grad=True)
    id,frac=swap_index(id,frac)
    normal[1]=interp_fn(id,frac,SDFGrid,grad=True)
    id,frac=swap_index(id,frac)
    normal[0]=interp_fn(id,frac,SDFGrid,grad=True)
    return normal/max(normal.norm(),eps)
def swap_index_batch(id,frac):
    nrX=len(id)
    idr=[[[None for k in range(nrX)] for j in range(nrX)] for i in range(nrX)]
    fracx,fracy,fracz=torch.split(frac,[1,1,1],dim=2)
    for i in range(nrX):
        for j in range(nrX):
            for k in range(nrX):
                idr[k][i][j]=id[i][j][k]
    return idr,torch.cat([fracz,fracx,fracy],dim=2)
def cal_norm_batch(id,frac,SDFGrid,interp_fn,eps=1e-5):
    nrN=id[0][0][0].shape[1]
    normal2=interp_fn(id,frac,SDFGrid,grad=True).view([-1,nrN,1])
    id,frac=swap_index_batch(id,frac)
    normal1=interp_fn(id,frac,SDFGrid,grad=True).view([-1,nrN,1])
    id,frac=swap_index_batch(id,frac)
    normal0=interp_fn(id,frac,SDFGrid,grad=True).view([-1,nrN,1])
    normal=torch.cat([normal0,normal1,normal2],dim=2)
    return normal/torch.clamp(torch.norm(normal,p=None,dim=2,keepdim=True),min=eps)

class SDF:
    def __init__(self,path,rad=0,res=64):
        if path is not None and os.path.exists(path+'.pickle'):
            tmp=pickle.load(open(path+'.pickle','rb'))
            self.path=tmp.path
            self.first=tmp.first
            self.last=tmp.last
            self.coef=tmp.coef
            self.max_limit=tmp.max_limit
            self.tensor=tmp.tensor
            return
        if rad>0:
            cellSz=rad*4/res
            self.path=path
            self.first=torch.tensor([cellSz,cellSz,cellSz])/2-rad*2
            self.last=-torch.tensor([cellSz,cellSz,cellSz])/2+rad*2
            self.coef=torch.tensor([1/cellSz,1/cellSz,1/cellSz])
            self.max_limit=torch.FloatTensor([res,res,res])-1.001
            self.tensor=torch.randn(res,res,res)
            for ii in range(res):
                for jj in range(res):
                    for kk in range(res):
                        pt=self.first+torch.tensor([float(ii),float(jj),float(kk)])*cellSz
                        self.tensor[ii][jj][kk]=torch.norm(pt)-rad
        else:
            reader=vtk.vtkStructuredPointsReader()
            reader.SetFileName(path)
            reader.ReadAllTensorsOn()
            reader.Update()
            data=reader.GetOutput()
            dim=data.GetDimensions()
            #interpolation info
            self.path=path
            self.first=torch.tensor(data.GetPoint(0))
            self.last=torch.tensor(data.GetPoint(data.GetNumberOfPoints()-1))
            self.coef=1/torch.tensor(data.GetSpacing())
            self.max_limit=torch.FloatTensor(dim)-1.001
            #read data
            self.tensor=torch.randn(dim[0],dim[1],dim[2])
            for ii in range(dim[0]):
                for jj in range(dim[1]):
                    for kk in range(dim[2]):
                        self.tensor[ii][jj][kk]=data.GetScalarComponentAsDouble(ii,jj,kk,0)
        if path is not None:
            pickle.dump(self,open(path+'.pickle','wb'))
        
    def writeVTK(self,path):
        imageData=vtk.vtkImageData()
        imageData.SetOrigin(self.first[0],self.first[1],self.first[2])
        imageData.SetSpacing(1/self.coef[0],1/self.coef[1],1/self.coef[2])
        imageData.SetDimensions(self.tensor.shape[0],self.tensor.shape[1],self.tensor.shape[2])
        if vtk.VTK_MAJOR_VERSION <= 5:
            imageData.SetNumberOfScalarComponents(1)
            imageData.SetScalarTypeToDouble()
        else: imageData.AllocateScalars(vtk.VTK_DOUBLE,1)
        dims=imageData.GetDimensions()
        # Fill every entry of the image data with "2.0"
        for ii in range(dims[2]):
            for jj in range(dims[1]):
                for kk in range(dims[0]):
                    imageData.SetScalarComponentFromDouble(ii,jj,kk,0,self.tensor[ii,jj,kk])
        #write
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(path)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInputConnection(imageData.GetProducerPort())
        else: writer.SetInputData(imageData)
        writer.Write()

class Distance(torch.nn.Module):
    def __init__(self):
        super(Distance,self).__init__()
    
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
        id=[[[None,None],[None,None]],[[None,None],[None,None]]]
        frac=torch.clamp(torch.min((pss-first)*coef,maxl.double()),min=0.0)
        id[0][0][0]=torch.stack((torch.floor(frac[:,:,0])  ,torch.floor(frac[:,:,1])  ,torch.floor(frac[:,:,2])  ),2)
        id[1][0][0]=torch.stack((torch.floor(frac[:,:,0])+1,torch.floor(frac[:,:,1])  ,torch.floor(frac[:,:,2])  ),2)
        id[0][1][0]=torch.stack((torch.floor(frac[:,:,0])  ,torch.floor(frac[:,:,1])+1,torch.floor(frac[:,:,2])  ),2)
        id[1][1][0]=torch.stack((torch.floor(frac[:,:,0])+1,torch.floor(frac[:,:,1])+1,torch.floor(frac[:,:,2])  ),2)
        id[0][0][1]=torch.stack((torch.floor(frac[:,:,0])  ,torch.floor(frac[:,:,1])  ,torch.floor(frac[:,:,2])+1),2)
        id[1][0][1]=torch.stack((torch.floor(frac[:,:,0])+1,torch.floor(frac[:,:,1])  ,torch.floor(frac[:,:,2])+1),2)
        id[0][1][1]=torch.stack((torch.floor(frac[:,:,0])  ,torch.floor(frac[:,:,1])+1,torch.floor(frac[:,:,2])+1),2)
        id[1][1][1]=torch.stack((torch.floor(frac[:,:,0])+1,torch.floor(frac[:,:,1])+1,torch.floor(frac[:,:,2])+1),2)
        frac=frac-id[0][0][0]
        dss=interp3d_batch(id,frac,sdfs)
        nss=cal_norm_batch(id,frac,sdfs,interp3d_batch)
        return dss,nss,torch.clamp(dss,max=0).sum()*(-1)
    
    def forward_validation(self,pss,sdfs):
        pss=pss.transpose(1,2)
        batch_size=pss.shape[0]
        assert batch_size == len(sdfs)
        dss=np.zeros((batch_size,pss.shape[1]))
        nss=np.zeros((batch_size,pss.shape[1],3))
        loss=0
        id=[[[None,None],[None,None]],[[None,None],[None,None]]]
        for i in range(batch_size):
            for j in range(pss.shape[1]):
                frac=torch.zeros(3)
                for d in range(3):
                    frac[d]=(pss[i,j,d]-sdfs[i].first[d])*sdfs[i].coef[d]
                    frac[d]=torch.clamp(frac[d],min=0.0,max=sdfs[i].max_limit[d])
                id[0][0][0]=torch.tensor([torch.floor(frac[0])  ,torch.floor(frac[1])  ,torch.floor(frac[2])  ])
                id[1][0][0]=torch.tensor([torch.floor(frac[0])+1,torch.floor(frac[1])  ,torch.floor(frac[2])  ])
                id[0][1][0]=torch.tensor([torch.floor(frac[0])  ,torch.floor(frac[1])+1,torch.floor(frac[2])  ])
                id[1][1][0]=torch.tensor([torch.floor(frac[0])+1,torch.floor(frac[1])+1,torch.floor(frac[2])  ])
                id[0][0][1]=torch.tensor([torch.floor(frac[0])  ,torch.floor(frac[1])  ,torch.floor(frac[2])+1])
                id[1][0][1]=torch.tensor([torch.floor(frac[0])+1,torch.floor(frac[1])  ,torch.floor(frac[2])+1])
                id[0][1][1]=torch.tensor([torch.floor(frac[0])  ,torch.floor(frac[1])+1,torch.floor(frac[2])+1])
                id[1][1][1]=torch.tensor([torch.floor(frac[0])+1,torch.floor(frac[1])+1,torch.floor(frac[2])+1])
                frac=frac-id[0][0][0]
                dss[i,j]=interp3d(id,frac,sdfs[i])
                nss[i,j,:]=cal_norm(id,frac,sdfs[i],interp3d)
                loss+=min(dss[i,j],0.0)
        return dss,nss,loss*(-1)

    def value_check(self,hand,sdfs):
        configurations=torch.randn(len(sdfs),hand.nr_dof()+6)
        pss,h_nss=hand.forward(configurations)
        dssRef,nssRef,lossRef=self.forward_validation(pss,sdfs)
        dss,nss,loss=self.forward(pss,sdfs)
        print('dss=',np.linalg.norm(dss.numpy()),'dssErr=',np.linalg.norm(dss.numpy()-dssRef))
        print('nss=',np.linalg.norm(nss.numpy()),'nssErr=',np.linalg.norm(nss.numpy()-nssRef))
        print('loss=',loss,'lossErr=',loss-lossRef)
        
    def grad_check(self,hand,sdfs):
        configurations=torch.randn(len(sdfs),hand.nr_dof()+6)
        pss,h_nss=hand.forward(configurations)
        pssTest=torch.randn(pss.shape)
        pssTest.requires_grad_()
        print('AutoGradCheck=',torch.autograd.gradcheck(self,(pssTest,sdfs),eps=1e-6,atol=1e-6,rtol=1e-5,raise_exception=True))

def test_SDF(sdf,type,interval=10):
    off=interval/sdf.coef
    pss=None
    #sdf write
    sdf.writeVTK('iso-surface.vti')
    #x
    x=sdf.first.tolist()[0]
    while x<sdf.last[0]:
        #y
        y=sdf.first.tolist()[1]
        while y<sdf.last[1]:
            #z
            z=sdf.first.tolist()[2]
            while z<sdf.last[2]:
                if pss is None:
                    pss=torch.tensor([x,y,z]).view(1,3,1)
                else:
                    new_pt=torch.tensor([x,y,z]).view(1,3,1)
                    pss=torch.cat([pss,new_pt],dim=2)
                z+=off[2]
            y+=off[1]
        x+=off[0]
    #normal
    fn=type()
    _,nss,_=fn.forward(pss,[sdf])
    nss=pss.transpose(1,2)+nss*interval/sdf.coef.view([1,1,3])
    #getRawData
    pss=pss.transpose(1,2).squeeze(0).tolist()
    nss=nss.squeeze(0).tolist()
    #construct VTK points
    Points=vtk.vtkPoints()
    for p in pss:
        Points.InsertNextPoint(p[0],p[1],p[2])
    for p in nss:
        Points.InsertNextPoint(p[0],p[1],p[2])
    #construct VTK lines
    Lines=vtk.vtkCellArray()
    for i in range(len(pss)):
        l=vtk.vtkLine()
        l.GetPointIds().SetId(0,i)
        l.GetPointIds().SetId(1,i+len(pss))
        Lines.InsertNextCell(l)
    #write to normal VTK
    polydata=vtk.vtkPolyData()
    polydata.SetPoints(Points)
    polydata.SetLines(Lines)
    polydata.Modified()
    if vtk.VTK_MAJOR_VERSION<=5:
        polydata.Update()
    writer=vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("normals.vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else: writer.SetInputData(polydata)
    writer.Write()
    exit()

if __name__=='__main__':
    #test sphere
    # sdf=SDF('sphere.vtk',rad=0.5)
    #test_SDF(sdf,type=Distance)
    sdf_list=[]
    filter_files('sdf','vtk',sdf_list)
    sdfs=[SDF(f) for f in sdf_list]
    if len(sdfs)==0:
        sdfs.append(SDF(None,rad=200.0))
    hand_paths=['hand/BarrettHand/','hand/ShadowHand/']
    for path in hand_paths:
        hand=Hand(path,1,True)
        my_distance=Distance()
        my_distance.value_check(hand,sdfs)
        my_distance.grad_check(hand,sdfs)