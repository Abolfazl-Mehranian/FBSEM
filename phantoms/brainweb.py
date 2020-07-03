"""
Created on May 2019
BrainWeb phantom library


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk

"""
import numpy as np
from phantoms.phantomlib import random_lesion, regrid, zero_pad
from sys import platform
import os

def PETbrainWebPhantom(phanPath, phantom_number=0,voxel_size=None,image_size=None, num_lesions = 10, \
                       lesion_size_mm = [2,10], pet_lesion = False,t1_lesion = False, t2_lesion = False,hot_cold_ratio = 0.5, return_hirez = False):

    if type(phantom_number)==list:
         pet = np.zeros([len(phantom_number),]+image_size)
         mumap = 0*pet
         t1 = 0*pet
         t2 = 0*pet
         for i in range(len(phantom_number)):
             pet[i,:,:,:], mumap[i,:,:,:], t1[i,:,:,:], t2[i,:,:,:] = PETbrainWebPhantom(phanPath, phantom_number[i], \
                voxel_size,image_size, num_lesions, lesion_size_mm, pet_lesion, t1_lesion, t2_lesion,hot_cold_ratio)
         return pet, mumap, t1, t2
    
    if voxel_size is None:
        voxel_size = [2.08625, 2.08625, 2.03125]
    if image_size is None:
        image_size = [344,344,127]
    #filename='D:\\pyTorch\\brainweb_20_raws\\subject_04.raws'
    filename = download_brain_web(phanPath, phantom_number)
    if filename.endswith('.gz'):
         import gzip
         file = gzip.open(filename, "r")
         phantom = np.frombuffer(file.read(), dtype='uint16').copy()
    else:
          phantom = np.fromfile(filename, dtype='uint16')
    phantom = phantom.reshape([362, 434, 362]).transpose(1,2,0)
    phantom =phantom[::-1, :, :]
    
    # PHANTOM PARAMETER
    indicesCsf = phantom == 16
    indicesWhiteMatter = phantom == 48
    indicesGrayMatter = phantom == 32
#    indicesFat = phantom == 64
#    indicesMuscleSkin = phantom == 80
    indicesSkin = phantom == 96
    indicesSkull = phantom == 112
#    indicesGliaMatter = phantom == 128
#    indicesConnectivity = phantom == 144
    indicesMarrow = phantom == 177
    indicesDura = phantom == 161
    indicesBone = indicesSkull | indicesMarrow | indicesDura
    indicesAir  = phantom ==0

    # 0=Background, 1=CSF, 2=Gray Matter, 3=White Matter, 4=Fat, 5=Muscle, 6=Muscle/Skin, 7=Skull, 8=vessels, 9=around fat, 10 =dura matter, 11=bone marrow
    mumap = np.zeros(phantom.shape,dtype='float')
    mu_bone_1_cm = 0.13;
    mu_tissue_1_cm = 0.0975;
    mumap[phantom >0] = mu_tissue_1_cm
    mumap[indicesBone] = mu_bone_1_cm
     
    #TRANSFORM THE ATANOMY INTO PET SIGNALS
    whiteMatterAct = 32
    grayMatterAct = 96
    skinAct = 16;
    pet = phantom;
    pet[indicesWhiteMatter] = whiteMatterAct
    pet[indicesGrayMatter] = grayMatterAct
    pet[indicesSkin] = skinAct 
    pet[~indicesWhiteMatter &  ~indicesGrayMatter & ~indicesSkin] = skinAct/2
    pet[indicesAir] = 0
    
    # T1
    t1 = 0*phantom;
    whiteMatterT1 = 154
    grayMatterT1 = 106
    skinT1 = 92
    skullT1 = 48
    marrowT1 = 180
    duraT1 = 48
    csfT2 = 48
    t1[indicesWhiteMatter] = whiteMatterT1
    t1[indicesGrayMatter] = grayMatterT1
    t1[indicesSkin] = skinT1
    t1[~indicesWhiteMatter & ~indicesGrayMatter & ~indicesSkin & ~indicesBone] = 0
    t1[indicesSkull] = skullT1
    t1[indicesMarrow] = marrowT1
    t1[indicesBone] = duraT1
    t1[indicesCsf] = csfT2
    # T2
    t2 = 0*phantom;
    whiteMatterT2 = 70;
    grayMatterT2 = 100;
    skinT2 = 70;
    skullT2 = 100;
    marrowT2 = 250;
    csfT2 = 250;
    duraT2 = 200;
    t2[indicesWhiteMatter] = whiteMatterT2
    t2[indicesGrayMatter] = grayMatterT2
    t2[indicesSkin] = skinT2
    t2[~indicesWhiteMatter & ~indicesGrayMatter & ~indicesSkin & ~indicesBone] = 0
    t2[indicesCsf] = csfT2
    t2[indicesSkull] = skullT2
    t2[indicesMarrow] = marrowT2
    t2[indicesBone] = duraT2
    
    if pet_lesion:
         lesion_pet = random_lesion(indicesWhiteMatter, num_lesions,lesion_size_mm)
         lesion_values = np.zeros(num_lesions)
         indx = list(range(num_lesions))
         np.random.shuffle(indx)
         split = int(np.floor(num_lesions*(hot_cold_ratio)))
         cold_idx,hot_idx = indx[split:],indx[:split]
         lesion_values[hot_idx] = grayMatterAct*1.5
         lesion_values[cold_idx] = whiteMatterAct*0.5
         for le in range(num_lesions):
              pet[lesion_pet[:,:,:,le]] = lesion_values[le]
    if t1_lesion:
         lesion_t1 = random_lesion(indicesWhiteMatter, num_lesions,lesion_size_mm)
         lesion_values = np.zeros(num_lesions)
         indx = list(range(num_lesions))
         np.random.shuffle(indx)
         split = int(np.floor(num_lesions*(hot_cold_ratio)))
         cold_idx,hot_idx = indx[split:],indx[:split]
         lesion_values[hot_idx] = whiteMatterT1*1.5
         lesion_values[cold_idx] = grayMatterT1*0.8
         for le in range(num_lesions):
              t1[lesion_t1[:,:,:,le]] = lesion_values[le]              
    if t2_lesion:
         lesion_t2 = random_lesion(indicesWhiteMatter, num_lesions,lesion_size_mm)
         lesion_values = np.zeros(num_lesions)
         indx = list(range(num_lesions))
         np.random.shuffle(indx)
         split = int(np.floor(num_lesions*(hot_cold_ratio)))
         cold_idx,hot_idx = indx[split:],indx[:split]
         lesion_values[hot_idx] = whiteMatterT1*1.5
         lesion_values[cold_idx] = grayMatterT1*0.8
         for le in range(num_lesions):
              t2[lesion_t2[:,:,:,le]] = lesion_values[le]  
    
    if return_hirez: pet_h = pet.copy()
    pet = regrid(pet,[0.5,0.5,0.5],voxel_size)
    pet = zero_pad(pet,image_size)
    pet[pet<0]=0
    if return_hirez: mumap_h = mumap.copy()
    mumap = regrid(mumap,[0.5,0.5,0.5],voxel_size)
    mumap = zero_pad(mumap,image_size)
    mumap[mumap<0]=0
    if return_hirez: t1_h = t1.copy()
    t1 = regrid(t1,[0.5,0.5,0.5],voxel_size)
    t1 = zero_pad(t1,image_size)
    t1[t1<0]=0
    if return_hirez: t2_h = t2.copy()
    t2 = regrid(t2,[0.5,0.5,0.5],voxel_size)
    t2 = zero_pad(t2,image_size)
    t2[t2<0]=0
    
    if return_hirez:
          return pet, mumap, t1, t2, pet_h, mumap_h, t1_h, t2_h
    else:
          return pet, mumap, t1, t2

def download_brain_web(phanPath, phantom_number = 0, download_all = False):
     # return file name of phantom_number (0:19), if dosn't exist download it
     if platform == "win32":
          bar = '\\'
     else:
          bar = '/'
     links=[
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject04_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject05_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject06_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject18_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject20_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject38_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject41_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject42_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject43_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject44_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject45_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject46_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject47_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject48_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject49_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject50_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject51_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject52_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject53_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
     'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject54_crisp&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D']
     if download_all:
          for i in range(len(links)): 
               download_brain_web(phanPath, phantom_number = i)
          return
     if phantom_number>19:
          raise ValueError("Choose a phantom number in [0, 19]")
     flname = phanPath +bar+ 'brainWeb_subject_'+ str(phantom_number)+'.raws.gz'
     if not os.path.isfile(flname):
        if not os.path.isdir(phanPath): os.makedirs(phanPath)
        import urllib
        urllib.request.urlretrieve(links[phantom_number], flname)

     return flname






