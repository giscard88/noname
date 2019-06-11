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
        flag_single=False
        if len(size)==1:
            norm=np.linalg.norm(roV)
            roV=roV/(norm+np.finfo(float).eps)
            flag_single=True   
        else:
            norm=np.linalg.norm(roV, axis=1)
            for xin in range(size[0]):

                roV[xin,:]=roV[xin,:]/(norm[xin]+np.finfo(float).eps)

        if self.blank:
            if len(size)==1:
                self.wm.append(roV)
                self.wm=np.array(self.wm)
                self.blank=False
            else:
                self.wm.append(roV[0,:])
                self.wm=np.array(self.wm)
                self.blank=False
                self.Mat_batch(roV[1:,:])

        else:
            self.Mat_batch(roV,flag=flag_single )

    def Mat_batch(self, roV, flag=False):            
        temp=np.matmul(self.wm,roV.T)
        if flag:
            if np.amax(temp)<self.threshold:
                self.wm=np.vstack((self.wm,roV))
        else:    
            print 'temp',temp
            max_vec=np.amax(temp,axis=0)
            print 'max_vec',max_vec
            idx=np.where(max_vec<self.threshold)[0]
            sel_vecs=roV[idx,:]
            print idx
            self.wm=np.vstack((self.wm,sel_vecs))


'''
mem=CogMem_numpy(5,0.8)
a=np.random.rand(5,5)



mem.Test_batch(a)
print '1', mem.wm
     
mem.Test_batch(a[2,:])
print '2',mem.wm

mem.Test_batch(np.random.rand(5))
print '3',mem.wm        

'''

# end of script





        
            
        
         
    

