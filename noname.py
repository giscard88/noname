import numpy as np
import json


class CogMem_numpy:
    def __init__(self,size, threshold):
        self.inS=size
        self.threshold=threshold
        self.wm=[]
        self.blank=True

    def Multiplication(self, roV):
        temp=np.matmul(self.wm,roV.T)
        idx=np.where(temp>self.threshold)[0]
        if len(idx)>0:
            return False
        else:
            return True

    def Test_i(self, roV):
        if self.blank:
            self.wm.append(roV)
            self.wm=np.array(self.wm)

            self.blank=False
        else:
            temp_flag=self.Multiplication(roV)
            if temp_flag:
               
                self.wm=np.vstack((self.wm,roV.T))

    def Test(self,roV):
        size=roV.shape
        if len(size)==1:
            vec_sel=roV
            norm=np.linalg.norm(vec_sel)
            vec_sel=vec_sel/norm            
            self.Test_i(vec_sel)
        else:
            for xin in range(r):
                vec_sel=roV[xin,:]
                norm=np.linalg.norm(vec_sel)
                vec_sel=vec_sel/norm            
                self.Test_i(vec_sel)

    def Test_batch(self,roV):
        size=roV.shape
        if len(size)==1:
            norm=np.linalg.norm(roV)
            roV=roV/(norm+np.finfo(float).eps)   
        else:
            norm=np.linalg.norm(roV, axis=1)
            for xi, xin in enumerate(range(size[1])):
                roV[xin,:]=roV[xin,:]/(norm[xi]+np.finfo(float).eps)

        if self.blank:
            if len(size)==1:
                self.wm.append(roV)
                self.wm=np.array(self.wm)
                self.blank=False
            else:
                self.wm.append(roV[0,:])
                self.wm=np.array(self.wm)
                self.blank=False

        else:
            temp=np.matmul(self.wm,roV.T)
            max_vec=np.amax(temp,axis=0)
            idx=np.where(max_vec)[0]
            sel_vecs=roV[idx,:]
            self.wm=np.vstack((self.wm,sel_vecs))
     
            
        






        
            
        
         
    

