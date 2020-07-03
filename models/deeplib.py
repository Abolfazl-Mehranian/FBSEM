"""
Created on April 2019
Deep learning reconstruction library


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk

"""


import numpy as np

def buildBrainPhantomDataset(PET, save_training_dir, phanPath, phanType = 'brainweb', phanNumber = None, is3d = True, num_rand_rotations=1, \
                             rot_angle_degrees=15, psf_hd=0.25, psf_ld=0.4, niter_hd = 15, niter_ld = 10, nsubs_hd = 14, nsubs_ld = 14, \
                                   counts_hd = 1e10, count_ld_window_3d=[90e6,120e6], count_ld_window_2d=[1e6,10e6], slices_2d = None, \
                                   pet_lesion =True, t1_lesion = True, num_lesions = 15, lesion_size_mm = [2,8], hot_cold_ratio = 0.6):

    # phanNumber: an numpy.array or list indicating number of phantoms, e.g. 10, np.arange(10,20) , [0,10,50] 
    """
    example:
    from numpy import arange
    from geometry.BuildGeometry_v4 import BuildGeometry_v4 
    from phantoms.deeplib import buildBrainPhantomDataset
    
    binPath = r'C:\MatlabWorkSpace\apirl-tags\APIRL1.3.3_win64_cuda8.0_sm35\build\bin'
    temPath = r'C:\pythonWorkSpace\tmp003'
    is3d = False
    phanType = 'brainkcl' #brainweb
    
    PET = BuildGeometry_v4('mmr',0.5)
    if is3d:
         PET.setApirlMmrEngine(binPath =binPath, temPath=temPath,gpu = True)
    else:
         PET.loadSystemMatrix(temPath,is3d=False)
         
    if phanType=='brainkcl':
       phanPath = r'J:\phantoms\brainKCL'
       save_training_dir = 'J:\\MoDL\\trainingDatasets\\brainKCL\\2D\\data02-test\\'
    else:
         phanPath = r'J:\phantoms\brainWeb'
         save_training_dir = 'J:\\MoDL\\trainingDatasets\\brainweb\\2D\\data01\\'

    phanNumber = arange(0,10,1)
 
    buildBrainPhantomDataset(PET, save_training_dir, phanPath, phanType =phanType,  phanNumber = phanNumber,is3d = is3d, count_ld_window_2d=[1e6,10e6])
    
    """
    bar = PET.engine.bar
    import os
    from phantoms.phantomlib import imRotation
    if phanType.lower()=='brainweb':
         from phantoms.brainweb import PETbrainWebPhantom
         if phanNumber is None:
              phanNumber = np.arange(20)
         else:
              if np.isscalar(phanNumber): phanNumber = [phanNumber]
              phanNumber = np.array(phanNumber)
    elif phanType.lower() == 'brainkcl':
         from phantoms.brainkcl import PETbrainKclPhantom
         
         phanFolder = os.listdir(phanPath)
         if phanNumber is None:
              phanNumber = np.arange(len(phanFolder))
         else:
              if np.isscalar(phanNumber): phanNumber = [phanNumber]
              phanNumber = np.array(phanNumber)
    if not os.path.isdir(save_training_dir):
         os.makedirs(save_training_dir)
    if slices_2d is None:
         slices_2d = np.arange(65,85,2)  #10 middle slices are chosen (contain mostly brain)  
    num_slices= len(slices_2d)
    
    if PET.is3d!=is3d:
         d = '3D' if is3d else '2D'
         raise ValueError(f" PET object is not {d}")
    voxel_size = np.array(PET.image.voxelSizeCm)*10
    image_size = PET.image.matrixSize
    
    dset = {'sinoLD':[],'imgLD':[], 'imgLD_psf':[],'imgHD':[],'AN':[],'RS':[],'imgGT':[],'mrImg':[], 'counts':[],'psf_hd':psf_hd,'psf_ld':psf_ld,'niter_hd': niter_hd, 'nsubs_hd': nsubs_hd, 
            'niter_ld': niter_ld, 'nsubs_ld': nsubs_ld,'num_lesions':num_lesions,'lesion_size_mm': lesion_size_mm, 'hot_cold_ratio': hot_cold_ratio,
            'rot_angle_degrees':rot_angle_degrees, 'counts_hd':counts_hd, 'count_ld_window_3d': count_ld_window_3d, 'count_ld_window_2d': count_ld_window_2d,'pet_lesion':pet_lesion,
            't1_lesion': t1_lesion, 'phanType':phanType, 'phanPath':[]}
    f=0
    for i in phanNumber:
        print(f"* create phantom {i}...")
        if phanType.lower()=='brainweb':
             img,mumap,t1,_= PETbrainWebPhantom(phanPath, i, voxel_size,image_size, num_lesions, lesion_size_mm, pet_lesion,t1_lesion, False,hot_cold_ratio)
        elif phanType.lower()=='brainkcl':
             nii_path = phanPath + bar + phanFolder[i] + bar + 'nii'
             dset['phanPath'] = nii_path
             img,mumap,t1 = PETbrainKclPhantom(nii_path,voxel_size,image_size, num_lesions, lesion_size_mm, pet_lesion ,t1_lesion, hot_cold_ratio)               
        randomAngle = 2*rot_angle_degrees*np.random.rand(num_rand_rotations)-rot_angle_degrees
        randomAngle[0]=0
        
        for j in range(num_rand_rotations):
            print(f"* {j} th rotation of phantom {i}...")
            if is3d: 
                 imgr = imRotation(img,randomAngle[j])
                 imgr[imgr<0]=0 
                 mumapr = imRotation(mumap,randomAngle[j])
                 mumapr[mumapr<0]=0
                 t1r = imRotation(t1,randomAngle[j])
                 t1r[t1r<0]=0
                 counts_ld = count_ld_window_3d[0] + (count_ld_window_3d[1]-count_ld_window_3d[0])*np.random.rand()
                 prompts_hd,AF,NF,_ = PET.simulateSinogramData(imgr, mumap=mumapr, counts=counts_hd, psf=psf_hd)
                 prompts_ld,_,_,_ =  PET.simulateSinogramData(imgr, AF=AF, NF = NF, counts=counts_ld, psf=psf_ld)
                 AN = AF*NF
                 img_hd = PET.OSEM3D(prompts_hd,AN=AN,niter=niter_hd, nsubs=nsubs_hd,psf=psf_hd)
                 img_ld = PET.OSEM3D(prompts_ld,AN=AN,niter=niter_ld,nsubs=nsubs_ld,psf=psf_hd)
                 img_ld_psf = PET.OSEM3D(prompts_ld,AN=AN,niter=niter_ld,nsubs=nsubs_ld,psf=psf_ld)

                 
                 dset['sinoLD'] = prompts_ld
                 dset['imgLD'] = img_ld
                 dset['imgLD_psf'] = img_ld_psf
                 dset['imgHD'] = img_hd
                 dset['AN'] = AN
                 dset['imgGT'] = imgr
                 dset['mrImg'] = t1r
                 dset['counts'] = counts_ld
                 
                 flname = save_training_dir+bar+'data-'
                 while os.path.isfile(flname+str(f)+'.npy'):
                      f+=1
                 np.save(flname+str(f)+'.npy', dset)
                 f+=1
            else: # num_slices of 3D phantoms are used to generate 2D phantoms
                 imgr = np.transpose(imRotation(img[:,:,slices_2d],randomAngle[j]),(2,0,1))
                 imgr[imgr<0]=0 
                 mumapr = np.transpose(imRotation(mumap[:,:,slices_2d],randomAngle[j]),(2,0,1))
                 mumapr[mumapr<0]=0
                 t1r = np.transpose(imRotation(t1[:,:,slices_2d],randomAngle[j]),(2,0,1))
                 t1r[t1r<0]=0

                 counts_ld = count_ld_window_2d[0] + (count_ld_window_2d[1]-count_ld_window_2d[0])*np.random.rand()
                 print("* simulate HD 2D sinograms...")
                 prompts_hd,AF,NF,_ = PET.simulateSinogramData(imgr, mumap=mumapr, counts=counts_hd, psf=psf_hd)
                 print("* simulate LD 2D sinograms...")
                 prompts_ld,_,_,_ =  PET.simulateSinogramData(imgr, AF=AF, NF = NF, counts=counts_ld, psf=psf_ld)
                 AN = AF*NF
                 print("* reconstruct HD 2D sinograms...")
                 img_hd = PET.OSEM2D(prompts_hd,AN=AN,niter=niter_hd, nsubs=nsubs_hd,psf=psf_hd)
                 print("* reconstruct LD 2D sinograms...")
                 img_ld = PET.OSEM2D(prompts_ld,AN=AN,niter=niter_ld,nsubs=nsubs_ld,psf=psf_hd)
                 print("* reconstruct LD+psf 2D sinograms...")
                 img_ld_psf = PET.OSEM2D(prompts_ld,AN=AN,niter=niter_ld,nsubs=nsubs_ld,psf=psf_ld)
                 print("* save datasets...")
                 for k in range(num_slices):
                      dset['sinoLD'] = prompts_ld[k,:,:]
                      dset['imgLD'] = img_ld[k,:,:]
                      dset['imgLD_psf'] = img_ld_psf[k,:,:]
                      dset['imgHD'] = img_hd[k,:,:]
                      dset['AN'] = AN[k,:,:]
                      dset['imgGT'] = imgr[k,:,:]
                      dset['mrImg'] = t1r[k,:,:]
                      dset['counts'] = counts_ld
                 
                      flname = save_training_dir+bar+'data-'
                      while os.path.isfile(flname+str(f)+'.npy'):
                           f+=1
                      np.save(flname+str(f)+'.npy', dset)
                      f+=1

"""
Data loader and Training models
"""

from torch.utils.data import Dataset
from numpy import load, ceil

class DatasetPetMr_v2(Dataset):
    def __init__(self, filename, num_train, transform=None, target_transform=None, is3d=False, imgLD_flname = None,crop_factor = 0, allow_pickle=True):
        """
        filename = ['save_dir,'prefix']
        num_train =number of traning datasets
        set "has_gtruth=False" for invivo data
        """
        self.transform = transform
        self.target_transform=target_transform
        self.is3d = is3d
        self.filename = filename
        self.num_train = num_train
        self.imgLD_flname = imgLD_flname
        self.crop_factor = crop_factor
        self.allow_pickle =allow_pickle
        
    def crop_sino(self,sino):
         if self.crop_factor!=0:
              i =  int(ceil(sino.shape[0]*self.crop_factor/2.0)*2)//2
              sinOut = sino[i:sino.shape[0]-i]
         else:
              sinOut = sino
         return sinOut
    def crop_img(self,img):
         if self.crop_factor!=0:
              i =  int(ceil(img.shape[0]*self.crop_factor/2.0)*2)//2    
              imgOut = img[i:img.shape[0]-i, i:img.shape[1]-i]
         else:
              imgOut = img
         return imgOut
       
    def __len__(self):
        return self.num_train
   
    def __getitem__(self, index):
        dset = load(self.filename[0]+self.filename[1]+str(index)+'.npy',allow_pickle=self.allow_pickle).item()
        
        sinoLD =  self.crop_sino(dset['sinoLD'])
        AN = self.crop_sino(dset['AN'])
        imgHD = self.crop_img(dset['imgHD'])
        mrImg = self.crop_img(dset['mrImg'])
        counts = dset['counts']

        if 'RS' in dset and type(dset['RS'])!=list:
             RS = self.crop_sino(dset['RS'])
        else:
             RS = 0        
        if 'imgGT' in dset and type(dset['imgGT'])!=list:
             imgGT = self.crop_img(dset['imgGT'])
        else:
             imgGT = 0
        if 'imgLD' in dset and type(dset['imgLD'])!=list:
            imgLD = self.crop_img(dset['imgLD'])
        elif self.imgLD_flname is not None:
             dset = load(self.imgLD_flname[0]+self.imgLD_flname[1]+str(index)+'.npy').item()
             imgLD = self.crop_img(dset['imgLD'])
        else:
             imgLD = 0
        if 'imgLD_psf' in dset  and type(dset['imgLD_psf'])!=list:
            imgLD_psf = self.crop_img(dset['imgLD_psf'])
        elif self.imgLD_flname is not None:
             dset = load(self.imgLD_flname[0]+self.imgLD_flname[1]+str(index)+'.npy').item()
             imgLD_psf = self.crop_img(dset['imgLD_psf'])
        else:
             imgLD_psf = 0
        if self.transform is not None:
            sinoLD = self.transform(sinoLD)
            AN = self.transform(AN)   
            if not np.isscalar(RS):
                 RS = self.transform(RS) 
        if self.target_transform is not None:
            imgHD = self.target_transform(imgHD)
            mrImg = self.target_transform(mrImg)
            if not np.isscalar(imgLD):
                 imgLD = self.target_transform(imgLD)
            if not np.isscalar(imgLD_psf):
                 imgLD_psf = self.target_transform(imgLD_psf)
            if not np.isscalar(imgGT):
                 imgGT = self.target_transform(imgGT)

        return sinoLD, imgHD, AN, RS,imgLD, imgLD_psf, mrImg, counts, imgGT,index
   

   
def train_test_split(dset, num_train, batch_size, test_size, valid_size=0, num_workers = 0, shuffle=True):
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader
    from numpy import floor, random

    indx = list(range(num_train))
    if shuffle:
        random.shuffle(indx)
    split = int(floor(num_train*(test_size)))
    train_idx,test_idx = indx[split:],indx[:split]
    
    valid_loader = None
    valid_idx = None
    if valid_size:
        if shuffle:
            random.shuffle(train_idx)
        split = int(floor(len(train_idx)*valid_size))
        train_idx,valid_idx = train_idx[split:],train_idx[:split]
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = DataLoader(dset,batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers, pin_memory=False)
  
    test_sampler = SubsetRandomSampler(test_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(dset,batch_size = batch_size, sampler = train_sampler, num_workers = num_workers, pin_memory=False)
    test_loader = DataLoader(dset,batch_size = batch_size, sampler = test_sampler, num_workers = num_workers,pin_memory=False)
 
    return train_loader, test_loader, valid_loader#, train_idx, test_idx, valid_idx


def PETMrDataset(filename, num_train, batch_size, test_size, valid_size=0, num_workers = 0, \
                 transform=None, target_transform=None, is3d=False, imgLD_flname = None,   shuffle=True, crop_factor = 0):
 
     dset = DatasetPetMr_v2(filename,num_train, transform, target_transform, is3d, imgLD_flname,crop_factor)
     train_loader, test_loader, valid_loader = train_test_split(dset, num_train, batch_size, test_size, valid_size, num_workers, shuffle)
     # for indecs call train_loader.sampler.indices
     return train_loader, valid_loader, test_loader


def noise_realizations(img, mumap, t1, save_dir, pet_geometry_dir=None, PET=None, psf_hd=0.25,psf_ld=0.4,\
                              niter_hd = 15, niter_ld = 10, nsubs_hd = 14, nsubs_ld = 14,\
                              counts_hd = 1e10,counts_ld=None, num_noise_realizations = 10):
     """
     example:
          
    from geometry.BuildGeometry_v4 import BuildGeometry_v4 
    from phantoms.brainkcl import PETbrainKclPhantom
    from models.deeplib import noise_realizations
    temPath = r'C:\pythonWorkSpace\tmp003'
    is3d = False
    phanType = 'brainkcl' #brainweb
    
    PET = BuildGeometry_v4('mmr',0.5)
    PET.loadSystemMatrix(temPath,is3d=False)          
          
    voxel_size = np.array(PET.image.voxelSizeCm)*10
    image_size = PET.image.matrixSize

    nii_path = r'J:\phantoms\brainKCL\add_mMR_62\nii'
    img,mumap,t1 = PETbrainKclPhantom(nii_path,voxel_size,image_size, num_lesions=20, lesion_size_mm=[2,8], pet_lesion=True ,t1_lesion=True, hot_cold_ratio=0.7)
          
          # pick up a slice (passing through pet lesions)
    idx = np.nonzero(img.max(axis=1).max(axis=0) == img.max())[0]
    slice = 74  
    counts_ld = 0.1e6    
    save_dir = r'J:\MoDL\trainingDatasets\brainKCL\2D\data02\noise-realization-add_mMR_62\noise-0.5e6'
    noise_realizations(img[:,:,slice], mumap[:,:,slice], t1[:,:,slice], save_dir, PET=PET, counts_ld=counts_ld)
          
     """
     import os
     import numpy as np 
         
     if not os.path.exists(save_dir):
          os.makedirs(save_dir)
     if PET is None:
          from geometry.BuildGeometry import BuildGeometry
          PET = BuildGeometry('mmr')
          PET.loadSystemMatrix(pet_geometry_dir)     

     if counts_ld is None:
          count_ld_window_2d=[1e6,10e6]
          counts_ld = count_ld_window_2d[0] + (count_ld_window_2d[1]-count_ld_window_2d[0])*np.random.rand()
     dset = {'sinoLD':[],'imgLD':[], 'imgLD_psf':[],'imgHD':[],'AN':[],'RS':[],'imgGT':[],'mrImg':[], 'counts':[],'psf_hd':psf_hd,'psf_ld':psf_ld,'niter_hd': niter_hd, 'nsubs_hd': nsubs_hd, 
            'niter_ld': niter_ld, 'nsubs_ld': nsubs_ld, 'counts_hd':counts_hd,'counts_ld': counts_ld}           

         
     prompts_hd,AF,NF,_ = PET.simulateSinogramData(img, mumap=mumap, counts=counts_hd, psf=psf_hd)
     AN = AF*NF
     img_hd = PET.OSEM2D(prompts_hd,AN=AN,niter=niter_hd, nsubs=nsubs_hd,psf=psf_hd)
     
     for j in range(num_noise_realizations):
          prompts_ld,_,_,_ =  PET.simulateSinogramData(img, AF=AF, NF = NF, counts=counts_ld, psf=psf_ld)
          img_ld = PET.OSEM2D(prompts_ld,AN=AN,niter=niter_ld,nsubs=nsubs_ld,psf=psf_hd)
          img_ld_psf = PET.OSEM2D(prompts_ld,AN=AN,niter=niter_ld,nsubs=nsubs_ld,psf=psf_ld)

          dset['sinoLD'] = prompts_ld
          dset['imgHD'] = img_hd
          dset['AN'] = AN
          dset['imgGT'] = img
          dset['mrImg'] = t1   
          dset['counts'] = counts_ld
          dset['imgLD'] = img_ld
          dset['imgLD_psf'] = img_ld_psf
          np.save(save_dir+'\data-nr'+str(j)+'.npy', dset)

        
def noise_levels_realizations(img, mumap, t1, save_dir, pet_geometry_dir=None, PET=None, psf_hd=0.15,psf_ld=0.15,\
                       counts_hd = 100e6,ld_count_window=[50e3,1e6],num_count_levels = 5, num_noise_realizations = 1):
     """
     example:
          from deeplib import noise_levels_realizations
          
          phantom_filename='D:\\pyTorch\\brainweb_20_raws\\subject_04.raws'
          save_dir = r'D:\pyTorch\brainweb_20_raws\subject_04_noise_lr'
          pet_geometry_dir='J:\\MyPyWorkSapce\\mmr2008_brain\\'
          
          pet, mumap, t1,_ = PETbrainWebPhantom(phantom_filename,pet_lesion = True,t1_lesion = True)
          
          # pick up a slice (passing through pet lesions)
          idx = np.nonzero(pet.max(axis=1).max(axis=0) == pet.max())[0]
          slice = 50
          noise_levels_realizations(pet[:,:,slice], mumap[:,:,slice], t1[:,:,slice], save_dir, PET=PET, psf_hd=0.15,psf_ld=0.15,\
                            counts_hd = 100e6,ld_count_window=[1e5,1e6],num_count_levels = 5, num_noise_realizations = 10)
          
     """
     import os
     import numpy as np 
         
     if not os.path.exists(save_dir):
          os.makedirs(save_dir)
     if PET is None:
          from geometry.BuildGeometry import BuildGeometry
          PET = BuildGeometry('mmr')
          PET.loadSystemMatrix(pet_geometry_dir)     
     
     dset = {'sinoLD':[],'imgHD':[],'AN':[],'imgGT':[],'mrImg':[], 'counts':[],'psf':psf_ld}
         
     prompts_hd,AF,_= PET.simulateDataBatch2D(img,mumap,counts=counts_hd,psf=psf_hd)
     prompts_hd = prompts_hd[None,:,:]
     AF = AF[None,:,:]
     img_hd = PET.OsemBatch2D(prompts_hd,AN=AF,psf=psf_hd,nsubs=14,niter=10)
            
     counts_ld = np.linspace(ld_count_window[0],ld_count_window[1],num_count_levels)
     
     for j in range(num_noise_realizations):
          for i in range(num_count_levels):
               prompts_ld,_,_= PET.simulateDataBatch2D(img,mumap,counts=counts_ld[i],psf=psf_ld)
               prompts_ld = prompts_ld[None,:,:]

               img_ld = PET.OsemBatch2D(prompts_ld,AN=AF,psf=psf_ld,nsubs=6,niter=10)
               dset['sinoLD'] = prompts_ld[0,:,:]
               dset['imgHD'] = img_hd[0,:,:]
               dset['AN'] = AF[0,:,:]
               dset['imgGT'] = img
               dset['mrImg'] = t1   
               dset['counts'] = counts_ld[i]
               dset['imgOsem'] = img_ld[0,:,:]
               np.save(save_dir+'\data-cl'+str(i)+'nr'+str(j)+'.npy', dset)

class dotstruct():
    def __setattr__(self, name, value):
         self.__dict__[name] = value
    def __getitem__(self, name):
        return self[name]
    def as_dict(self):
        dic = {}
        for item in self.__dict__.keys(): 
             dic[item] = self.__dict__.get(item)
        return dic

def setOptions(arg,opt,trasnfer=True):
    # update common items of arg from opt
    for item in arg.__dict__.keys(): 
        if item in opt.__dict__.keys():
            arg.__dict__[item] = opt.__dict__.get(item)
    # trasnfer unique items of opt into arg
    if trasnfer:
        for item in opt.__dict__.keys():
            if item not in arg.__dict__.keys():
                arg.__dict__[item] = opt.__dict__.get(item)
    return arg


def imShowBatch(x,batch_size=None, is3d = False, slice_num = None, vmax=None,cmap=None,title=None,coronal=False,figsize=(20,10),caption=None,rotation=0):
    from matplotlib import pyplot as plt
    import numpy as np
    if batch_size is None:
         batch_size = x.shape[0]
    if cmap is None:
        cmap = ['gist_yarg']*batch_size
    if type(cmap)!=list:
        cmap = [cmap]*batch_size

    if vmax is None:
        vmax = [None]*batch_size
    elif np.isscalar(vmax):
        vmax = [vmax]*batch_size
            
    fig, ax = plt.subplots(1,batch_size, sharex=True, sharey=True,figsize=figsize)
    for i in range(batch_size):
        if is3d:
            if coronal:
                if slice_num is None: slice_num = x.shape[1]//2
                img = np.rot90(x[i,slice_num,:,:],1)
            else:
                if slice_num is None: slice_num = x.shape[3]//2
                img = x[i,:,:,slice_num]
        else:
             img = x[i,:,:]
        if batch_size>1:
            ax[i].imshow(img, vmin =0,vmax = vmax[i],cmap=cmap[i]),ax[i].axis('off')
        else:
            ax.imshow(img, vmin =0,vmax = vmax[i],cmap=cmap[i]),ax.axis('off')
        if caption is not None:
            if batch_size>1:
                ax[i].set_title(caption[i],fontsize=22,va='bottom',rotation = rotation)
            else:
                ax.set_title(caption[i],fontsize=22,va='bottom',rotation = rotation)
    if title is not None:
        fig.suptitle(title,fontsize=15)
    fig.subplots_adjust(hspace=0,wspace=0)
    plt.pause(0.1)
    
def gaussFilterBatch(img,voxelSizeCm, fwhm, is3d=True):
    from scipy import ndimage
    fwhm = np.array(fwhm)
    if np.all(fwhm==0):
        return img

    if fwhm.shape==1:
        if is3d:
            fwhm=fwhm*np.ones([3,])
        else:
            fwhm=fwhm*np.ones([2,])
    if not is3d:
        voxelSizeCm = voxelSizeCm[0:2]
    sigma=fwhm/voxelSizeCm/np.sqrt(2**3*np.log(2))
        
    Filter = lambda x,sigma: ndimage.filters.gaussian_filter(x,sigma)
    if is3d:
         if np.ndim(img)==3:
            imOut = Filter(img,sigma)  
         else:
              imOut = 0*img
              for b in range(img.shape[0]):
                   imOut[b,:,:,:] = Filter(img[b,:,:,:],sigma)
    else:
         if np.ndim(img)==2:
            imOut = Filter(img,sigma)  
         else:
              imOut = 0*img
              for b in range(img.shape[0]):
                   imOut[b,:,:] = Filter(img[b,:,:],sigma)     
    return imOut

def zeroNanInfs(x):
     from torch import is_tensor,isnan,isinf, Tensor
     if is_tensor(x):
          x.data[isnan(x)]=Tensor([0])
          x.data[isinf(x)]=Tensor([0])
     else:
          x[np.isnan(x)]=0
          x[np.isinf(x)]=0
     return x


def crop(img, crop_factor=0, is3d = False): 
    # img = Tensor: (nBatch,nChannels,W,H,D), (nBatch,nChannels,W,H)
    # img = numpy.array: (nBatch,W,H,D), (nBatch,W,H), (W,H,D),  (W,H)
    
    # crop img in W,H dimenstions by crop_factor [0,1) %
    if crop_factor!=0:
         from torch import is_tensor
         from numpy import ceil
         isTensor = is_tensor(img)
         round_int = lambda x: int(ceil(x/2.0)*2)
         if isTensor:
             i = round_int(img.shape[2]*(crop_factor))//2
             j = round_int(img.shape[3]*(crop_factor))//2
             imgOut = img[:,:,i:img.shape[2]-i, j:img.shape[3]-j]
         else:
              if img.ndim==4 or (img.ndim==3 and not is3d):
                   i = round_int(img.shape[1]*(crop_factor))//2
                   j = round_int(img.shape[2]*(crop_factor))//2
                   imgOut = img[:,i:img.shape[1]-i, j:img.shape[2]-j]
              elif img.ndim==2 or (img.ndim==3 and is3d):
                   i = round_int(img.shape[0]*(crop_factor))//2
                   j = round_int(img.shape[1]*(crop_factor))//2
                   imgOut = img[i:img.shape[0]-i, j:img.shape[1]-j]
    else: 
        imgOut = img 
    return imgOut  

def uncrop(img,W,H=None,is3d = False):
    # img = Tensor: (nBatch,nChannels,W0,H0,D), (nBatch,nChannels,W0,H0)
    # img = numpy.array: (nBatch,W0,H0,D), (nBatch,W0,H0), (W0,H0,D),  (W0,H0)     
    
    # zero pads (W0,H0) dims of img to (W,H)
    from torch import is_tensor
    
    if H is None: H = W
    if is_tensor(img):
         if (img.shape[2]!=W or img.shape[3]!=H):
              from torch import zeros
              i = (W - img.shape[2])//2 
              j = (H - img.shape[3])//2 
              dims = [img.shape[0],img.shape[1],W,H]
              if img.dim()==5: dims.append(img.shape[4])
              imgOut = zeros(dims,dtype=img.dtype,device=img.device)
              imgOut[:,:,i:W-i, j:H-j] = img
         else:
             imgOut = img 
    else: 
         from numpy import zeros
         if img.ndim==4 and (img.shape[1]!=W or img.shape[2]!=H): #(nBatch,W,H,D)
              i = (W - img.shape[1])//2 
              j = (H - img.shape[2])//2 
              imgOut = zeros((img.shape[0],W,H,img.shape[3]),dtype=img.dtype)
              imgOut[:,i:W-i, j:H-j,:] = img
         elif (img.ndim==3 and not is3d) and (img.shape[1]!=W or img.shape[2]!=H): #(nBatch,W,H)
              i = (W - img.shape[1])//2 
              j = (H - img.shape[2])//2 
              imgOut = zeros((img.shape[0],W,H),dtype=img.dtype)
              imgOut[:,i:W-i, j:H-j] = img 
         elif (img.ndim==3 and is3d) and (img.shape[0]!=W or img.shape[1]!=H): #(W,H,D)
              i = (W - img.shape[0])//2 
              j = (H - img.shape[1])//2 
              imgOut = zeros((W,H,img.shape[2]),dtype=img.dtype)
              imgOut[i:W-i, j:H-j,:] = img 
         elif img.ndim==2  and (img.shape[0]!=W or img.shape[1]!=H): #(W,H)
              i = (W - img.shape[0])//2 
              j = (H - img.shape[1])//2 
              imgOut = zeros((W,H),dtype=img.dtype)
              imgOut[i:W-i, j:H-j] = img  
         else:
              imgOut = img 
    return imgOut 

def toNumpy(x):
    return x.detach().cpu().numpy().astype('float32')