'''
    This file is modified from http:\\
'''


import cvxpy as cp
import numpy as np
import random,math,torch
import time
# torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)


def cmpToKey(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self,obj,*args):
            self.obj=obj
        def __lt__(self,other):
            return mycmp(self.obj,other.obj)<0
        def __gt__(self,other):
            return mycmp(self.obj,other.obj)>0
        def __eq__(self,other):
            return mycmp(self.obj,other.obj)==0
        def __le__(self,other):
            return mycmp(self.obj,other.obj)<=0  
        def __ge__(self,other):
            return mycmp(self.obj,other.obj)>=0
        def __ne__(self,other):
            return mycmp(self.obj,other.obj)!=0
    return K


class Directions:
    def __init__(self,res=4,dim=6):
        self.dirs=[]
        self.res=res
        self.dim=dim
        n=np.array([0.0]*dim,dtype=np.float64)
        for d in range(dim):
            n[d]= 1
            self.addDirs(0,d,n)
            n[d]=-1
            self.addDirs(0,d,n)
        #sort dir
        def cmp(A,B):
            for d in range(dim):
                if A[d]<B[d]:
                    return -1
                elif A[d]>B[d]:
                    return 1
            return 0
        self.dirs=sorted(self.dirs,key=cmpToKey(cmp))
        #make compact
        j=0
        for i in range(len(self.dirs)):
            if i>0 and (self.dirs[i]==self.dirs[i-1]).all():
                continue
            else: 
                self.dirs[j]=self.dirs[i]
                j+=1
        self.dirs=self.dirs[0:j]
        #normalize
        for i in range(len(self.dirs)):
            self.dirs[i]/=np.linalg.norm(self.dirs[i])
    def addDirs(self,d0,d,n):
        if d0==self.dim:
            self.dirs.append(n.astype(np.float64))
        elif d0==d:
            self.addDirs(d0+1,d,n)
        else:
            for i in range(self.res):
                n[d0]=-1+2*i/float(self.res-1)
                self.addDirs(d0+1,d,n)
    def printDirs(self):
        print('res=%d dim=%d #dir=%d'%(self.res,self.dim,len(self.dirs)))
        #for d in self.dirs:
        #    print(d)


def cross(p):
    ret=np.zeros((3,3),dtype=np.float64)
    ret[2,1]= p[0]
    ret[1,2]=-p[0]
    ret[0,2]= p[1]
    ret[2,0]=-p[1]
    ret[1,0]= p[2]
    ret[0,1]=-p[2]
    return ret


def support(M,mu,alpha,pss,dss,nss,s,beta=None,hand_normal=None):
    w=cp.Variable(6)
    fss=[]
    cons=[]
    sumFN=0
    sumW=-w
    for i in range(len(pss)):
        fss.append(cp.Variable(3))
        fN=fss[-1].T@nss[i]
        fT=fss[-1]-fN*nss[i]
        #normal sum
        if hand_normal is not None:
            sumFN+=fN*math.exp(alpha*abs(dss[i])+beta*(1+np.dot(nss[i],hand_normal[i])))
        else:
            sumFN+=fN*math.exp(alpha*abs(dss[i]))
        #frictional cone
        cons.append(cp.SOC(fN*mu,fT))
        #sum of f
        f2w=np.concatenate((np.eye(3,3,dtype=np.float64),cross(pss[i])))
        sumW+=f2w@fss[-1]
    #normal sum
    cons.append(sumFN<=1)
    #sum of f
    cons.append(sumW==0)
    #objective
    if M is not None:
        obj=cp.Maximize(w.T@M@s)
    else: obj=cp.Maximize(w.T@s)
    prob=cp.Problem(obj,cons)
    prob.solve()
    return obj.value


def support_analytic(M,mu,alpha,pss,dss,nss,s,beta=None,hand_normal=None):
    support_j=None
    for i in range(len(pss)):
        f2w=np.concatenate((np.eye(3,3,dtype=np.float64),cross(pss[i])))
        if M is not None:
            f2w=np.matmul(M,f2w)
        w_perp=(np.mat(s)*f2w*np.mat(nss[i]).T)[0,0]
        w_para=np.linalg.norm(np.mat(s)*f2w*(np.identity(3)-np.mat(nss[i]).T*np.mat(nss[i])))
        if mu*w_perp>w_para:
            max_sw=(w_perp+w_para**2/w_perp)
        else: max_sw=max(0,w_perp+mu*w_para)
        if hand_normal is not None:
            max_sw*=np.exp(-alpha*abs(dss[i])-beta*(1+np.dot(nss[i],hand_normal[i])))
        else:
            max_sw*=np.exp(-alpha*abs(dss[i]))
        support_j=max_sw if support_j is None else max(support_j,max_sw)
    return support_j


def compute_Q1(M,mu,alpha,pss,dss,nss,sss,beta=None,hand_normal=None,analytic=False):
    f_support=support_analytic if analytic else support
    ret=None
    for s in sss:
        if ret is None:
            if hand_normal is not None:
                ret=f_support(M,mu,alpha,pss,dss,nss,s,beta=beta,hand_normal=hand_normal)
            else:
                ret=f_support(M,mu,alpha,pss,dss,nss,s)
        else:
            if hand_normal is not None:
                ret=min(ret,f_support(M,mu,alpha,pss,dss,nss,s,beta=beta,hand_normal=hand_normal))
            else:
                ret=min(ret,f_support(M,mu,alpha,pss,dss,nss,s))
    return ret


class ComputeQ1Layer(torch.nn.Module):
    def __init__(self):
        super(ComputeQ1Layer,self).__init__()
        self.crossx=torch.tensor([[ 0, 0, 0],
                                  [ 0, 0,-1.],
                                  [ 0, 1.,0]]).view(1,3,3)
        self.crossy=torch.tensor([[ 0, 0, 1.],
                                  [ 0, 0, 0],
                                  [-1.,0, 0]]).view(1,3,3)
        self.crossz=torch.tensor([[ 0,-1., 0],
                                  [ 1.,0, 0],
                                  [ 0, 0, 0]]).view(1,3,3)
        self.identity_f2w=torch.tensor([[ 1.,0, 0],
                                        [ 0, 1.,0],
                                        [ 0, 0, 1.],
                                        [ 0, 0, 0],
                                        [ 0, 0, 0],
                                        [ 0, 0, 0]])
    def forward(self,M,mu,alpha,pss,dss,nss,sss,beta=None,hand_normal=None):
        #M: [6,6]
        #mu: scalar
        #alpha: scalar
        #pss: [b,3,np]
        #dss: [b,np]
        #nss: [b,np,3]
        #sss: [nd,6]
        pss=pss.transpose(1,2)
        np=pss.shape[1]
        nd=sss.shape[0]
        #cross product
        pssx,pssy,pssz=torch.split(pss,[1,1,1],dim=2)
        pssc =pssx.view([-1,np,1,1])*self.crossx.type(pssx.type())
        pssc+=pssy.view([-1,np,1,1])*self.crossy.type(pssx.type())
        pssc+=pssz.view([-1,np,1,1])*self.crossz.type(pssx.type())
        #f2w
        f2w=torch.nn.functional.pad(pssc,[0,0,3,0])+self.identity_f2w.type(pssx.type())
        #M*f2w
        if M is not None:
            f2w=torch.matmul(M.type(f2w.type()),f2w)
        #w_perp
        f2w_n=torch.matmul(f2w,nss.view([-1,np,3,1])).view([-1,np,1,6,1])
        w_perp=torch.matmul(sss.view([nd,1,6]).type(f2w_n.type()),f2w_n).view([-1,np,nd])
        #w_para
        f2w_Innt=f2w.view([-1,np,1,6,3])-torch.matmul(f2w_n,nss.view([-1,np,1,1,3]))
        w_para=torch.matmul(sss.view([nd,1,6]).type(f2w_Innt.type()),f2w_Innt).view([-1,np,nd,3])
        w_para=torch.norm(w_para,p=None,dim=3)
        #support analytic
        cond=w_perp*mu>w_para
        in_cone=w_perp+w_para**2/w_perp
        not_in_cone=torch.clamp(w_perp+mu*w_para,min=0)
        support=torch.where(cond,in_cone,not_in_cone)
        #exponential decay
        if hand_normal is not None:
            hand_normal=hand_normal.transpose(1,2)
            bs=nss.shape[0]
            hand_point_num=nss.shape[1]
            normal_dot=torch.bmm(nss.view(-1,3).unsqueeze(1),hand_normal.contiguous().view(-1,3).unsqueeze(-1))
            normal_dot=normal_dot.view(bs,hand_point_num)
            support*=torch.exp(torch.abs(dss)*-alpha-beta*(1+normal_dot)).view([-1,np,1])
        else:
            support*=torch.exp(torch.abs(dss)*-alpha).view([-1,np,1])
        #Q1
        Q1,max_index=torch.max(support,dim=1)
        # softmin=torch.nn.Softmin(dim=1)
        # Q1=softmin(Q1).mean()
        Q1,min_index=torch.min(Q1,dim=1)
        print("Max support index: "+str(max_index[0][min_index[0]].item()))
        return Q1


if __name__ == '__main__':
    dirs=Directions(res=2)
    M = None
    # pss = np.array([[-1, -1, -1],[-1, -1, 1],[-1, 1, -1],[-1, 1, 1],
    #                 [1, -1, -1],[1, -1, 1],[1, 1, -1],[1, 1, 1]])/1.73
    # nss = np.array([[-1, -1, -1],[-1, -1, 1],[-1, 1, -1],[-1, 1, 1],
    #                 [1, -1, -1],[1, -1, 1],[1, 1, -1],[1, 1, 1]])/1.73
    # dss = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    pss = np.array([[-1, -1, -1],[-1, 1, -1],[1, 1, -1]])/1.73
    nss = np.array([[-1, -1, -1],[-1, 1, -1],[1, 1, -1]])/1.73
    dss = np.array([0, 0, 0])
    # pss = np.array([[-1, -1, -1],[1,-1,1],[1,1,-1],[-1,1,1]])/1.73
    # nss = np.array([[-1, -1, -1],[1,-1,1],[1,1,-1],[-1,1,1]])/1.73
    # dss = np.array([0, 0, 0, 0])
    start = time.time()
    Q1 = compute_Q1(M,0.7,0.1,pss,dss,nss,dirs.dirs,analytic=True)
    end = time.time()
    print("Q1:\t", Q1)
    print("Time:\t", end-start)

