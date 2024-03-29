import numpy as np
import json
import torch


class CogMem_numpy:
    '''
    synaptic weight-based memory implemented using numpy. Support for labels is not currently available yet. 
    synapses are added to store novel inputs 
    '''
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

    def Projection(self,roV):
        self.image=np.matmul(self.wm,roV.T)

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

class CogMem_torch:
    '''
    synaptic weight-based memory implemented using pytorch. Labels can also be forwarded to be stored.
    synapses are added to store novel inputs 
    '''
    def __init__(self,size, threshold):
        self.inS=size
        self.threshold=threshold
        self.wm=[]
        self.blank=True
        self.labels_=[]

    def Test_batch(self,roV, label=None):
        size=roV.size()
        self.label=label
        print size
        flag_single=False
        if len(size)==1:
            roV=roV.view(-1,self.inS)
            norm=torch.norm(roV)
            roV=roV/(norm+np.finfo(float).eps)
            flag_single=True   
        else:
            norm=torch.norm(roV, dim=1)
            for xin in range(size[0]):
                roV[xin,:]=roV[xin,:]/(norm[xin]+np.finfo(float).eps)

        if self.blank:
            if len(size)==1:
                self.wm=roV
                if self.label is not None:
                     self.labels_.append(self.label)
                else:
                     self.labels_.append(-9999)
                self.blank=False
            else:
                
                self.wm=roV[0,:]
                if self.label is not None:
                     self.labels_.append(self.label[0])
                else:
                     self.labels_.append(-9999)
                self.blank=False
                self.Mat_batch(roV[1:,:],self.label[1:])

        else:
            self.Mat_batch(roV,self.label,flag=flag_single)

    def Mat_batch(self, roV, label_,flag=False):

        if flag:
            roV_T=torch.transpose(roV,-1,0)
            #print roV            
            temp=torch.matmul(self.wm,roV_T)
            if torch.max(temp).item()<self.threshold:
                self.wm=torch.cat((self.wm,roV),dim=0)
                if label_ is not None:
                    self.labels_.append(label_)
                else:
                    self.labels_.append(-9999) # -9999 represents a null label. 
        else:

            roV_T=torch.transpose(roV,0,1)
          
            temp=torch.matmul(self.wm,roV_T)    
            if len(temp.size())==1:
                max_vec=temp.numpy()
            else:
                max_vec=torch.max(temp,dim=0)[0].numpy()

            idx=np.where(max_vec<self.threshold)[0]

            idx_t=torch.from_numpy(idx)
            sel_vecs=roV[idx_t,:]
            self.wm=self.wm.view(-1,self.inS)

            
            self.wm=torch.cat((self.wm,sel_vecs),0)
            if label_ is not None:
                for l in idx:
                    self.labels_.append(label_[l])
            else:
                for l in idx:
                    self.labels_.append(-9999)

    def Projection(self, roV): # roV is expected to be a matrix, not a single vector

        roV_T=torch.transpose(roV,0,1)

        self.image=torch.matmul(self.wm,roV_T)

class CogMem_label_torch:
    '''
    synaptic weight-based memory implemented using pytorch. Labels can also be forwarded to be stored. 
    synapses are added when inputs are novel (according to the cosine similarity or when labels are novel. 

    '''
    def __init__(self,size, threshold):
        self.inS=size
        self.threshold=threshold
        self.wm=[]
        self.blank=True
        self.labels_=[]

    def Test_batch(self,roV, label=None):
        size=roV.size()
        self.label=label
        print size
        flag_single=False
        if len(size)==1:
            roV=roV.view(-1,self.inS)
            norm=torch.norm(roV)
            roV=roV/(norm+np.finfo(float).eps)
            flag_single=True   
        else:
            norm=torch.norm(roV, dim=1)
            for xin in range(size[0]):
                roV[xin,:]=roV[xin,:]/(norm[xin]+np.finfo(float).eps)

        if self.blank:
            if len(size)==1:
                self.wm=roV
                if self.label is not None:
                     self.labels_.append(self.label)
                else:
                     self.labels_.append(-9999)
                self.blank=False
            else:
                
                self.wm=roV[0,:]
                if self.label is not None:
                     self.labels_.append(self.label[0])
                else:
                     self.labels_.append(-9999)
                self.blank=False
                self.Mat_batch(roV[1:,:],self.label[1:])

        else:
            self.Mat_batch(roV,self.label,flag=flag_single)

    def Mat_batch(self, roV, label_,flag=False):

        if flag:
            roV_T=torch.transpose(roV,-1,0)
            #print roV            
            temp=torch.matmul(self.wm,roV_T)
            if torch.max(temp).item()<self.threshold:
                self.wm=torch.cat((self.wm,roV),dim=0)
                if label_ is not None:
                    self.labels_.append(label_)
                else:
                    self.labels_.append(-9999) # -9999 represents a null label. 
            elif label_ not in self.labes_:
                self.wm=torch.cat((self.wm,roV),dim=0)
                if label_ is not None:
                    self.labels_.append(label_)
                else:
                    self.labels_.append(-9999) # -9999 represents a null label.                   
        else:

            roV_T=torch.transpose(roV,0,1)
          
            temp=torch.matmul(self.wm,roV_T)    
            if len(temp.size())==1:
                max_vec=temp.numpy()
            else:
                max_vec=torch.max(temp,dim=0)[0].numpy()

            idx=np.where(max_vec<self.threshold)[0]

            idx_t=torch.from_numpy(idx)
            sel_vecs=roV[idx_t,:]
            sel_vecs=sel_vecs.view(-1,self.inS)
            #self.wm=self.wm.view(-1,self.inS)

            
            self.wm=torch.cat((self.wm,sel_vecs),0)
            if label_ is not None:
                for l in idx:
                    self.labels_.append(label_[l])
            else:
                for l in idx:
                    self.labels_.append(-9999)
            for li, l in labels_:
                if l not in self.labels_:
                    sel_vecs=roV[li,:]
                    sel_vecs=sel_vecs.view(-1,self.inS)    
                    self.wm=torch.cat((self.wm,sel_vecs),0)
                    if label_ is not None:
                        self.labels_.append(label_[li])
                    else:
                        self.labels_.append(-9999)

                

    def Projection(self, roV): # roV is expected to be a matrix, not a single vector

        roV_T=torch.transpose(roV,0,1)

        self.image=torch.matmul(self.wm,roV_T)


mem=CogMem_torch(5,0.9)
a=np.random.rand(5,5)
a=torch.from_numpy(a)


mem.Test_batch(a,[1,2,3,4,5])
print '1', mem.wm
print mem.labels_
     
mem.Test_batch(a[2,:],2)
print '2',mem.wm
print mem.labels_
mem.Test_batch(torch.from_numpy(np.random.rand(5)),6)
print '3',mem.wm        
print mem.labels_

a=np.random.rand(5,5)
a=torch.from_numpy(a)


mem.Test_batch(a,[11,12,13,14,15])
print '4', mem.wm
print mem.labels_


# end of script





        
            
        
         
    

