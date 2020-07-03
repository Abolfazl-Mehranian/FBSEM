
"""
Created on April 2019
Deep learning reconstruction library


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk

"""
import torch
import torch.nn as nn
from models.deeplib import zeroNanInfs, crop, uncrop
import numpy as np




class ResUnit_v2(nn.Module):
    def __init__(self, depth, num_kernels, kernel_size,in_channels,is3d):
        super(ResUnit_v2, self).__init__()
        self.in_channels =in_channels
        self.relu = nn.ReLU(inplace=True)
        layers = []
        if is3d:
             layers.append(nn.Conv3d(in_channels, num_kernels, kernel_size, padding=1))
             layers.append(nn.BatchNorm3d(num_kernels))
             layers.append(nn.ReLU(inplace=True))
             for _ in range(depth-2):
                  layers.append(nn.Conv3d(num_kernels, num_kernels, kernel_size, padding=1))
                  layers.append(nn.BatchNorm3d(num_kernels))
                  layers.append(nn.ReLU(inplace=True))
             layers.append(nn.Conv3d(num_kernels, 1, kernel_size, padding=1))
             layers.append(nn.BatchNorm3d(1))             
        else:
             layers.append(nn.Conv2d(in_channels, num_kernels, kernel_size, padding=1))
             layers.append(nn.BatchNorm2d(num_kernels))
             layers.append(nn.ReLU(inplace=True))
             for _ in range(depth-2):
                  layers.append(nn.Conv2d(num_kernels, num_kernels, kernel_size, padding=1))
                  layers.append(nn.BatchNorm2d(num_kernels))
                  layers.append(nn.ReLU(inplace=True))
             layers.append(nn.Conv2d(num_kernels, 1, kernel_size, padding=1))
             layers.append(nn.BatchNorm2d(1))
        self.dcnn = nn.Sequential(*layers)

    def forward(self, x, y=None):
        identity = x
        if y is not None:
            x = torch.cat((x,y),dim=1)
        out = self.dcnn(x)
        out += identity
        out = self.relu(out)
        return out
   
class FBSEMnet_v3(nn.Module):
    def __init__(self, depth,num_kernels, kernel_size, in_channels=1,is3d=False, reg_ccn_model = 'resUnit'):
        super(FBSEMnet_v3,self).__init__()
        if reg_ccn_model.lower()=='resUnit'.lower():
            self.regularize = ResUnit_v2(depth, num_kernels, kernel_size,in_channels,is3d)
        else:
            raise ValueError('model unkown')
        self.gamma = nn.Parameter(torch.rand(1),requires_grad=True)
        self.is3d = is3d
        
    def forward(self,PET,prompts,img=None,RS=None, AN=None, iSensImg = None, mrImg=None, niters = 10, nsubs=1, tof=False, psf=0,device ='cuda', crop_factor = 0):
         # e.g. crop_factor = 0.667
         
         batch_size = prompts.shape[0]
         device = torch.device(device)
         matrixSize = PET.image.matrixSize
         if 0<crop_factor<1: 
             Crop    = lambda x: crop(x,crop_factor,is3d=self.is3d)
             unCrop  = lambda x: uncrop(x,matrixSize[0],is3d=self.is3d)
         else:
             Crop    = lambda x: x
             unCrop  = lambda x: x         
         if self.is3d:  
             toTorch = lambda x: zeroNanInfs(Crop(torch.from_numpy(x).unsqueeze(1).to(device=device, dtype=torch.float32)))
             toNumpy = lambda x: zeroNanInfs(unCrop(x)).detach().cpu().numpy().squeeze(1).astype('float32')
             if iSensImg is None:
                  iSensImg = PET.iSensImageBatch3D(AN, nsubs, psf).astype('float32') 
             if img is None:
                  img =  np.ones([batch_size,matrixSize[0],matrixSize[1],matrixSize[2]],dtype='float32')
             if batch_size ==1:
                 if iSensImg.ndim==4:  iSensImg = iSensImg[None].astype('float32') 
                 if img.ndim==3:  img = img[None].astype('float32')
         else:
             reShape = lambda x: x.reshape([batch_size,matrixSize[0],matrixSize[1]],order='F')
             Flatten = lambda x: x.reshape([batch_size,matrixSize[0]*matrixSize[1]],order='F')
             toTorch = lambda x: zeroNanInfs(Crop(torch.from_numpy(reShape(x)).unsqueeze(1).to(device=device, dtype=torch.float)))
             toNumpy = lambda x: zeroNanInfs(Flatten((unCrop(x)).detach().cpu().numpy().squeeze(1)))
             if iSensImg is None:
                  iSensImg = PET.iSensImageBatch2D(AN, nsubs, psf) 
             if img is None:
                  img =  np.ones([batch_size,matrixSize[0]*matrixSize[1]],dtype='float32')  
             if batch_size ==1:
                 if iSensImg.ndim==2:  iSensImg = iSensImg[None].astype('float32') 
                 if img.ndim==1:  img = img[None].astype('float32')
         if mrImg is not None:
              mrImg = Crop(mrImg)
         imgt = toTorch(img)

         for i in range(niters):
              for s in range(nsubs):
                   if self.is3d:
                        img_em = img*PET.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, s, psf)*iSensImg[:,s,:,:,:]
                        img_emt = toTorch(img_em) 
                        img_regt = zeroNanInfs(self.regularize(imgt,mrImg)) 
                        S = toTorch(iSensImg[:,s,:,:,:])
                   else:
                        img_em = img*PET.forwardDivideBackwardBatch2D(img, prompts, RS, AN, nsubs, s, tof, psf)*iSensImg[:,s,:]
                        img_emt = toTorch(img_em) 
                        img_regt = zeroNanInfs(self.regularize(imgt,mrImg)) 
                        S = toTorch(iSensImg[:,s,:])
                   imgt = 2*img_emt/((1 - self.gamma*S*img_regt) + torch.sqrt((1 - self.gamma*S*img_regt)**2 + 4*self.gamma*S*img_emt)) 
                   img = toNumpy(imgt)
                   del img_em, img_emt, img_regt, S
         del iSensImg, prompts, RS, AN, PET, img

         return unCrop(imgt)

def Trainer(PET, model, opts, train_loader, valid_loader=None):
    from models.deeplib import dotstruct, setOptions, imShowBatch,crop
    import torch.optim as optim
    import os
    
    g = dotstruct()
    g.psf_cm = 0.15
    g.niters = 10
    g.nsubs = 6
    g.lr = 0.001
    g.epochs = 100
    g.in_channels = 1
    g.save_dir = os.getcwd()
    g.model_name = 'fbsem-pm-01'
    g.display = True
    g.disp_figsize=(20,10)
    g.save_from_epoch = None
    g.crop_factor = 0.3
    g.do_validation = True
    g.device = 'cpu'
    g.mr_scale = 5

    g = setOptions(g,opts)

    if not os.path.exists(g.save_dir):
        os.makedirs(g.save_dir)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=g.lr )
    toNumpy = lambda x: x.detach().cpu().numpy().astype('float32')

    train_losses = []
    valid_losses = []
    gamma = []
    
    for e in range(g.epochs):
         
         running_loss = 0
         for sinoLD, imgHD, AN, _,_, _, mrImg, _, _,index in train_loader: 
             #torch.cuda.empty_cache()
             AN=toNumpy(AN)
             RS = None
             sinoLD=toNumpy(sinoLD)
             imgHD = imgHD.to(g.device,dtype=torch.float32).unsqueeze(1)
    
             if g.in_channels==2:
                  mrImg = g.mr_scale*mrImg/mrImg.max()
                  mrImg = mrImg.to(g.device,dtype=torch.float32).unsqueeze(1)
             else:
                  mrImg = None
             optimizer.zero_grad()
             img = model.forward(PET,prompts = sinoLD,AN=AN, mrImg = mrImg,\
                                 niters=g.niters, nsubs = g.nsubs, psf=g.psf_cm, device=g.device, crop_factor=g.crop_factor)#, 
             loss = loss_fn(img,imgHD)
             loss.backward()
             optimizer.step()
             running_loss+=loss.item()
             
             if torch.isnan(model.gamma) or model.gamma.data<0:
                 model.gamma.data= model.gamma.data= torch.Tensor([0.01]).to(g.device,dtype=torch.float32)
             if display:
                 imShowBatch(crop(toNumpy(img).squeeze(),0.3), figsize = g.disp_figsize)
                 gam = model.gamma.clone().detach().cpu().numpy()[0]
                 print(f"gamma: {gam}")
                 gamma.append(gam)
             del sinoLD, AN, RS, mrImg, index
    
         else:
             train_losses.append(running_loss/len(train_loader))
             print(f"Epoch: {e+1}/{g.epochs}, Training loss: {train_losses[e]:.3f}")
             if g.do_validation:
                 valid_loss = 0
                 with torch.no_grad():
                     model.eval()
                     for sinoLD, imgHD, AN, _,_, _, mrImg, _, _,index in valid_loader:
                         AN=toNumpy(AN)
                         RS = None
                         sinoLD=toNumpy(sinoLD)
                         imgHD = imgHD.to(g.device,dtype=torch.float32).unsqueeze(1)
                         if g.in_channels==2:
                             mrImg = g.mr_scale*mrImg/mrImg.max()
                             mrImg = mrImg.to(g.device,dtype=torch.float32).unsqueeze(1)
                         else:
                             mrImg = None
                         img = model.forward(PET,prompts = sinoLD,AN=AN, mrImg = mrImg, niters=g.niters, nsubs = g.nsubs, psf=g.psf_cm, device=g.device, crop_factor=g.crop_factor)#, 
                         valid_loss +=loss_fn(img,imgHD).item()
                 valid_losses.append(valid_loss/len(valid_loader))
                 model.train()
                 print(f"Epoch: {e+1}/{g.epochs}, Validation loss: {valid_losses[e]:.3f}")
             if ((g.save_from_epoch is not None) and (g.save_from_epoch <=e)) or e==(g.epochs - 1):
                  g.state_dict = model.state_dict()
                  g.train_losses = train_losses
                  g.valid_losses = valid_losses
                  g.training_idx = train_loader.sampler.indices
                  g.gamma = gamma
                  
                  checkpoint = g.as_dict()
                  torch.save(checkpoint,g.save_dir+g.model_name+'-epo-'+str(e)+'.pth')

             torch.cuda.empty_cache()  
             
             
def fbsemInference(dl_model_flname, PET, sinoLD, AN, mrImg, niters=None, nsubs = None, device='cpu'):

    toNumpy = lambda x: x.detach().cpu().numpy().astype('float32')

    g = torch.load(dl_model_flname, map_location=torch.device(device))
    
    model = FBSEMnet_v3(g['depth'], g['num_kernels'], g['kernel_size'], g['in_channels'], g['is3d'], g['reg_ccn_model']).to(device)
    model.load_state_dict(g['state_dict'])
    
    AN=toNumpy(AN)
    RS = None
    sinoLD = toNumpy(sinoLD)

    if g['in_channels']==2:
         mrImg = g['mr_scale']*mrImg/mrImg.max()
         mrImg = mrImg.to(device,dtype=torch.float32).unsqueeze(1)
    else:
         mrImg = None
    niters = niters or g['niters']
    nsubs = nsubs or g['nsubs']
        
    with torch.no_grad():
        model.eval()
        img = model.forward(PET,prompts = sinoLD,AN=AN, mrImg = mrImg,\
                        niters=niters, nsubs = nsubs, psf=g['psf_cm'], device=device, crop_factor=g['crop_factor'])#, 
    return toNumpy(img).squeeze()