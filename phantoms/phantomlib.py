"""
Created on May 2019
BrainWeb phantom library


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk

"""
import numpy as np

def random_lesion(img, num_lesions,lesion_size_mm, voxel_radious_mm = 0.5):
     import random
     #img : indicesWhiteMatter

     idx = np.array(np.nonzero(img.flatten('F').reshape(-1,))).reshape(-1,)   
     i = random.sample(range(0, idx.shape[0]), num_lesions)
     i,j,k=col2ijk(idx[i],img.shape[0],img.shape[1],img.shape[2])

     x = np.arange(img.shape[0])
     y = np.arange(img.shape[1])
     z = np.arange(img.shape[2])
     xx, yy,zz = np.meshgrid(x, y,z,indexing='ij')
     
     r = lesion_size_mm[0]/voxel_radious_mm+(lesion_size_mm[1]-lesion_size_mm[0])/voxel_radious_mm**np.random.rand(num_lesions)
     lesions = np.zeros((img.shape[0],img.shape[1],img.shape[2],num_lesions))

     for le in range(num_lesions):
          lesions[:,:,:,le] = (((xx-i[le])**2+(yy-j[le])**2+(zz-k[le])**2)<r[le]**2).astype('float') 

     return lesions>0 

def col2ijk(m,Nx,Ny,Nk):
     n = Nx*Ny
     m+=1
     if np.max(m) > n**2:
        raise ValueError("m is greater than the max number of elements")
     k = np.ceil(m/n)
     temp = (m -(k-1)*n)
     j = np.ceil(temp/Nx)
     i = temp -(j-1)*Nx
     return i.astype(int)-1,j.astype(int)-1,k.astype(int)-1

def regrid(img, voxel_size, new_voxel_size, method='linear'):
    # img (numpy.ndarray)   
    # method: linear/ nearest
    from scipy.interpolate import RegularGridInterpolator
    
    new_shape = [int(np.round(img.shape[k]*voxel_size[k]/new_voxel_size[k])) for k in range(np.ndim(img))]
    if np.ndim(img)==3:
        x, y, z = [voxel_size[k] * np.arange(img.shape[k]) for k in range(3)]         
        f = RegularGridInterpolator((x, y, z), img, method=method)
        x, y, z = [np.linspace(0, voxel_size[k] * (img.shape[k]-1), new_shape[k]) for k in range(3)]
        new_grid = np.array(np.meshgrid(x, y, z, indexing='ij'))
        new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))
    else:
        x, y = [voxel_size[k] * np.arange(img.shape[k]) for k in range(2)]  
        f = RegularGridInterpolator((x, y), img, method=method)
        x, y = [np.linspace(0, voxel_size[k] * (img.shape[k]-1), new_shape[k]) for k in range(2)]
        new_grid = np.array(np.meshgrid(x, y, indexing='ij'))
        new_grid = np.moveaxis(new_grid, (0, 1, 2), (2, 0, 1))   

    new_img = f(new_grid)
    return new_img

def zero_pad(x,new_size):
    #new_size -->tuple     
    def idxs(m,q):
        if m <q:
            i = (q-m)/2
            ii=[int(i),int(q-i)]
        else:
            ii=[0,q]
        return ii
    
    X = np.zeros(new_size,dtype=x.dtype)
    # if new_size is smaller than x.shape, crop the image x
    if np.ndim(x)==3:
         i,j,k = [idxs(new_size[k],x.shape[k]) for k in range(3)] 
         x = x[i[0]:i[1],j[0]:j[1],k[0]:k[1]] 
    else:
         i,j = [idxs(new_size[k],x.shape[k]) for k in range(2)]
         x = x[i[0]:i[1],j[0]:j[1]] 
    # if new_size is larger than cropped x.shape
    if np.ndim(x)==3 and x.shape[2]>1:
        i,j,k = [idxs(x.shape[k],new_size[k]) for k in range(3)]
        X[i[0]:i[1],j[0]:j[1],k[0]:k[1]] = x
    else:
        i,j = [idxs(x.shape[k],new_size[k]) for k in range(2)]
        X[i[0]:i[1],j[0]:j[1]] = x
    return X

def imRotation(img,angle,num_rand_rotations=0):

     # example: imRotation(img,15), imRotation(img,15,5), imRotation(img,[3,45,-10])
    from scipy.ndimage import rotate
    
    if num_rand_rotations>0:
        # take angle as an interval
        Angle = 2*angle*np.random.rand(num_rand_rotations)-angle
        if num_rand_rotations>1:
             Angle[0] = 0 # to include no rotation
    else:
         if np.isscalar(angle):
              Angle = [angle]
         else:
              Angle = angle     

    imgr = np.zeros((len(Angle),) + img.shape,dtype=img.dtype)
    for a in range(len(Angle)):
         if Angle[a]!=0:
             if np.ndim(img)==3:
                 for i in range(img.shape[2]):
                     imgr[a,:,:,i] = rotate(img[:,:,i],Angle[a],reshape=False,order=1)
             else:
                 imgr[a,:,:] = rotate(img,Angle[a],reshape=False,order=1)
         else:
              if np.ndim(img)==3:
                   imgr[a,:,:,:] = img
              else:
                   imgr[a,:,:] = img    
    return np.squeeze(imgr)