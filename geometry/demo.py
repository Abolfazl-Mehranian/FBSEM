"""
Created on May 2019
PET image reconstruction demo


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk

"""
import numpy as np
from matplotlib import pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from phantoms.brainweb import PETbrainWebPhantom

temPath = r'C:\pythonWorkSpace\tmp'
geoPath = r'C:\pythonWorkSpace\tmp00'
phanPath = r'E:\PET-M\Phantoms\Brainweb'

radialBinCropFactor = 0.6
PET = BuildGeometry_v4('mmr',radialBinCropFactor)

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

# 2D OSEM 
img_osem_2d = PET.OSEM2D(y, AN=AF, niter = 10, nsubs = 6, psf= 0.2)
img_osem_2d_batch = PET.OSEM2D(y_batch, AN=AF_batch, niter = 10, nsubs = 6, psf= 0.2)







