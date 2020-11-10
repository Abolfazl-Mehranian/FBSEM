# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:55:55 2019

@author: abm15
"""
import numpy as np
from matplotlib import pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from phantoms.brainweb import PETbrainWebPhantom

binPath = r'C:\MatlabWorkSpace\apirl-tags\APIRL1.3.3_win64_cuda8.0_sm35\build\bin'
temPath = r'C:\pythonWorkSpace\tmp'
geoPath = r'C:\pythonWorkSpace\tmp00'
phanPath = r'E:\PET-M\Phantoms\Brainweb'

radialBinCropFactor = 0.6
PET = BuildGeometry_v3('mmr',radialBinCropFactor)

img_3d_batch, mumap_3d_batch, t1_3d_batch, _ = PETbrainWebPhantom(phanPath, phantom_number=[0,2,10], voxel_size= np.array(PET.image.voxelSizeCm)*10, \
                                           image_size=PET.image.matrixSize, pet_lesion = False, t1_lesion = False)
# 2D PET --------------------------------------------------
PET.loadSystemMatrix(geoPath,is3d=False)

img_2d = img_3d_batch[0,:,:,50]
mumap_2d = mumap_3d_batch[0,:,:,50]
img_2d_batch = img_3d_batch[:,:,:,50]
mumap_2d_batch = mumap_3d_batch[:,:,:,50]
psf_cm = 0.4

## 2D forward project
#y = PET.forwardProjectBatch2D(img_2d, psf = psf_cm)
#y_batch = PET.forwardProjectBatch2D(img_2d_batch, psf = psf_cm)

# simulate 2D noisy sinograms
y,AF,_ = PET.simulateSinogramData(img_2d, mumap = mumap_2d, counts= 1e6, psf = psf_cm)
y_batch,AF_batch,_ = PET.simulateSinogramData(img_2d_batch, mumap = mumap_2d_batch, counts= 1e6,  psf = psf_cm)

# 2D OSEM reconstruction
img_osem_2d = PET.OSEM2D(y, AN=AF, niter = 10, nsubs = 6, psf= 0.2)
img_osem_2d_batch = PET.OSEM2D(y_batch, AN=AF_batch, niter = 10, nsubs = 6, psf= 0.2)


# 3D PET ---------------------------------------------------
#PET.plotLorsAxialCoor()
#PET.plotMichelogram()
PET.setApirlMmrEngine(binPath =binPath, temPath=temPath,gpu = True)

# 3D forward project
#y3d = PET.forwardProjectBatch3D(img_3d_batch[0,:,:,:],psf=psf_cm)
#y3d_batch = PET.forwardProjectBatch3D(img_3d_batch,psf=psf_cm)

# simulate 3D noisy sinograms
y3d,AF3d,_ = PET.simulateSinogramData(img_3d_batch[0,:,:,:], mumap = mumap_3d_batch[0,:,:,:], counts= 100e6, psf = psf_cm)
y3d_batch,AF3d_batch,_ = PET.simulateSinogramData(img_3d_batch, mumap = mumap_3d_batch, counts= 100e6, psf = psf_cm)

# 3D OSEM reconstruction
img_osem_3d = PET.OSEM3D(y3d, AN=AF3d, niter = 3, nsubs = 21, psf=0.2)
img_osem_3d_batch = PET.OSEM3D(y3d_batch, AN=AF3d_batch, niter = 1, nsubs = 6, psf=0.2)


# switch back to 2D 
PET.is3d = False




