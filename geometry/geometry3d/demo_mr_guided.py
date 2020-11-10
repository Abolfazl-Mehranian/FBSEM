# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:21:09 2019

@author: abm15
"""
import numpy as np
from matplotlib import pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
binPath = r'C:\MatlabWorkSpace\apirl-tags\APIRL1.3.3_win64_cuda8.0_sm35\build\bin'
temPath = r'C:\pythonWorkSpace\tmp009'
PET = BuildGeometry_v4('mmr',0.5)
PET.setApirlMmrEngine(binPath =binPath, temPath=temPath,gpu = True)
PET.removeSampleFiles()
prompts, AN, RS = PET.get_e7sino(r'E:\PET-M\[18F]FDG_PET_MR_EEG_T1_MUMAP_PETRAW\mMR_BR1_051_anon\PET_raw_via_archive-Converted\PET_raw_via_archive-LM-00\sino')

img_osem = PET.OSEM3D(prompts, AN=AN, RS=RS, niter = 10, nsubs = 14, psf=0.45)
img_osem_s = PET.gaussFilterBatch(img_osem,0.5)
import nibabel as nib

mrImg_ = nib.load(r'E:\PET-M\[18F]FDG_PET_MR_EEG_T1_MUMAP_PETRAW\mMR_BR1_051_anon\MRI_via_scanner\T1\nii\mrInPet.nii')

mrImg = mrImg_.get_fdata()
mrImg = np.rot90(np.flip(mrImg,axis=2),1)
mrImg = PET.crop_img(mrImg,0.5)


#% %
from geometry.Prior import Prior
prior = Prior(PET.image.matrixSize, sWindowSize=3,imageCropFactor=[7])
prior_img_s = PET.gaussFilterBatch(mrImg/mrImg.max(),0.25)

weights = prior.BowshserWeights(prior_img_s,prior.nS//2)

iSensImg = PET.iSensImageBatch3D(AN,nsubs = 14, psf=0.45) 
img_mapem = PET.MAPEM3D(prompts, AN = AN, RS = RS,niter = 15,beta=40, iSensImg=iSensImg,nsubs = 14, psf=0.45, display = True, prior_object = prior,  weights = weights)
img_mapem_s = PET.gaussFilterBatch(img_mapem,0.25)
##%%
#i = 40
#l=0.55
#fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
#ax[0,0].imshow(prior_img_s[:,:,i],cmap='Greys_r')
#ax[0,1].imshow(img_osem_s[:,:,i],cmap='gist_heat',vmax=l)
#ax[0,2].imshow(img_mapem_s[:,:,i],cmap='gist_heat',vmax=l)
#i = 75
#ax[1,0].imshow(np.rot90(prior_img_s[:,i,:],-1),cmap='Greys_r')
#ax[1,1].imshow(np.rot90(img_osem_s[:,i,:],-1),cmap='gist_heat',vmax=l)
#ax[1,2].imshow(np.rot90(img_mapem_s[:,i,:],-1),cmap='gist_heat',vmax=l)
#%%
save_dir = r'E:\PET-M\[18F]FDG_PET_MR_EEG_T1_MUMAP_PETRAW\mMR_BR1_051_anon\PET_raw_via_archive-Converted\PET_raw_via_archive-LM-00\PET_raw_via_archive-00-PSF_000_000.v-DICOM\nii'
img_mapem = np.flip(np.rot90(PET.uncrop_img(img_mapem_s),-1),axis=2)
img_mapem = nib.Nifti1Image(img_mapem, mrImg_.affine, nib.Nifti1Header())
nib.save(img_mapem, save_dir+'\mapem-b40-15i-14s-045psf-025mm.nii')

img_osem_ = np.flip(np.rot90(PET.uncrop_img(img_osem),-1),axis=2)
img_osem_ = nib.Nifti1Image(img_osem_, mrImg_.affine, nib.Nifti1Header())
nib.save(img_osem_, save_dir+'\osem-10i-14s-045psf.nii')



