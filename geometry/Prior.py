"""
Created on Feb 2019
Image regularisation library


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk

"""

import numpy as np 
 
 
class Prior(object): 
         
    def __init__(self,imageSize, sWindowSize=3, lWindowSize=1, imageCropFactor=[0]): 
         
        self.imageSize = imageSize 
        if len(self.imageSize)==2: self.imageSize.append(1) 
        self.imageCropFactor = imageCropFactor 
        if np.mod(sWindowSize,2): 
            self.sWindowSize = sWindowSize 
        else: 
            raise ValueError("search window size must be odd") 
        if np.mod(lWindowSize,2): 
            self.lWindowSize = lWindowSize 
        else: 
            raise ValueError("local window size must be odd") 
        self.is3D = 1 if imageSize[2]>1 else 0  
        self.nS = sWindowSize**3 if self.is3D else sWindowSize**2 
        self.nL = lWindowSize**3 if self.is3D else lWindowSize**2         
        _,self.imageSizeCrop= self.imCrop()  
        self.SearchWindow, self.Wd = self.__neighborhood(self.sWindowSize) 
        self.LocalWindow,_ = self.__neighborhood(self.lWindowSize) 
 
    def __neighborhood(self,w): 
         
        n = self.imageSizeCrop[0] 
        m = self.imageSizeCrop[1] 
        h = self.imageSizeCrop[2] 
        wlen = 2*np.floor(w/2) 
        widx = xidx = yidx = np.arange(-wlen/2,wlen/2+1) 
 
        if h==1: 
            zidx = [0] 
            nN = w*w 
        else: 
            zidx = widx 
            nN = w*w*w 
         
        Y,X,Z = np.meshgrid(np.arange(0,m), np.arange(0,n), np.arange(0,h))                 
        N = np.zeros([n*m*h, nN],dtype='int32') 
        D = np.zeros([n*m*h, nN],dtype='float') 
        l = 0 
        for x in xidx: 
            Xnew = self.__setBoundary(X + x, n) 
            for y in yidx: 
                Ynew = self.__setBoundary(Y + y, m) 
                for z in zidx: 
                    Znew = self.__setBoundary(Z + z, h) 
                    N[:,l] = (Xnew + (Ynew)*n + (Znew)*n*m).reshape(-1,1).flatten('F') 
                    D[:,l] = np.sqrt(x**2+y**2+z**2) 
                    l += 1 
        eps = 1e-5
        D = 1/(D+eps) 
        D[D==1/eps]= 0 
        D = D/(np.sum(D,axis=1).reshape(-1,1)+eps)
        D[D==1/eps]= 0
        return N, D 
     
    def __setBoundary(self,X,n): 
        idx = X<0 
        X[idx] = X[idx]+n 
        idx = X>n-1 
        X[idx] = X[idx]-n 
        return X.flatten('F') 
 
    def imCrop(self,img=None): 
        if np.any(self.imageCropFactor): 
            if len(self.imageCropFactor)==1: 
                self.imageCropFactor = self.imageCropFactor*3 
            I = 0 
            if self.imageCropFactor[0]: 
                self.imageCropFactor[0] = np.max([2.5, self.imageCropFactor[0]]) 
                I = np.floor(self.imageSize[0]/self.imageCropFactor[0]).astype('int') 
            J = 0 
            if self.imageCropFactor[1]: 
                self.imageCropFactor[1] = np.max([2.5, self.imageCropFactor[1]]) 
                J = np.floor(self.imageSize[1]/self.imageCropFactor[1]).astype('int') 
            K = 0 
            if self.imageCropFactor[2] and self.is3D: 
                self.imageCropFactor[2] = np.max([2.5, self.imageCropFactor[2]]) 
                K = np.floor(self.imageSize[2]/self.imageCropFactor[2]).astype('int')             
            imageSizeCrop = [np.arange(I,self.imageSize[0]-I).shape[0], 
                             np.arange(J,self.imageSize[1]-J).shape[0], 
                             np.arange(K,self.imageSize[2]-K).shape[0]] 
            if img is not None: 
                if self.is3D: 
                    img = img[I:self.imageSize[0]-I, J:self.imageSize[1]-J, K:self.imageSize[2]-K]   
                else: 
                    img = img[I:self.imageSize[0]-I, J:self.imageSize[1]-J]  
        else: 
            imageSizeCrop = self.imageSize 
        return img,imageSizeCrop  
 
    def imCropUndo(self,img): 
        if np.any(self.imageCropFactor): 
            tmp = img.reshape(self.imageSizeCrop,order='F') 
            img = np.zeros(self.imageSize,tmp.dtype) 
            I = (self.imageSize[0] - self.imageSizeCrop[0])//2 
            J = (self.imageSize[1] - self.imageSizeCrop[1])//2 
            K = (self.imageSize[2] - self.imageSizeCrop[2])//2
            
            if self.is3D: 
                img[I:self.imageSize[0]-I, J:self.imageSize[1]-J, K:self.imageSize[2]-K] = tmp 
            else: 
                img[I:self.imageSize[0]-I, J:self.imageSize[1]-J] = tmp 
        return img 
     
    def Grad(self,img): 
        img,_ = self.imCrop(img) 
        img = img.flatten('F') 
        imgGrad = img.reshape(-1,1) - img[self.SearchWindow] 
        imgGrad[np.isnan(imgGrad)] = 0 
        return imgGrad 
     
    def GradT(self,imgGrad): 
        dP = np.sum(imgGrad,axis=1) 
        dP = dP.reshape(self.imageSizeCrop,order='F') 
        dP = self.imCropUndo(dP) 
        dP[np.isnan(dP)] = 0 
        dP = np.squeeze(dP)
        return dP 
     
    def Div(self,img): 
        img,_ = self.imCrop(img) 
        img = img.flatten('F') 
        imgDiv = img[self.SearchWindow] + img.reshape(-1,1) 
        imgDiv[np.isnan(imgDiv)] = 0 
        return imgDiv 
     
    def gaussianWeights(self,img,sigma): 
        w = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*self.Grad(img)**2/sigma**2)
        #w/=w.sum(axis=1).reshape(-1,1)
        return w
     
    def BowshserWeights(self,img,b): 
        if b>self.nS: 
            raise ValueError("Number of most similar voxels must be smaller than number of voxels per neighbourhood") 
        imgGradAbs = np.abs(self.Grad(img)) 
        Wb = 0*imgGradAbs 
        for i in range(imgGradAbs.shape[0]): 
            idx = np.argsort(imgGradAbs[i,:]) 
            Wb[i,idx[0:b]]=1 
        return Wb 