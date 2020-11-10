# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:07:04 2019

@author: abm15
"""

mct = {'model_number':          1104,
       'circularGantry':        1,
       'nBuckets':              48,
       'nBlockRings':           4,
       'nBlockPerRing':         48,
       'nPhysCrystalsPerBlock': 13,
       'useVirtualCrystal':     1,
       'detectorRadiusCm':      42.76,
       'sinogramDOIcm':         0.67,
       'LORDOIcm':              0.96,
       'nRadialBins':           400,
       'nMash':                 2,
       'rCrystalDimCm':         2.0,
       'xCrystalDimCm':         0.40728,
       'zCrystalDimCm':         0.4050,
       'transaxialFovCm':       69.7266,
       'span':                  11,
       'nSegments':             9,
       'maxRingDiff':           49,
       'nTofBins':              13,
       'coinciWindowWidthNsec': 4.0625,
       'tofResolutionNsec':     0.580,
       'tofOffsetNsec':         0.039            
       }

mmr = {'model_number':          2008,
       'circularGantry':        1,
       'nBuckets':              224,
       'nBlockRings':           8,
       'nBlockPerRing':         56,
       'nPhysCrystalsPerBlock': 8,
       'useVirtualCrystal':     1,
       'detectorRadiusCm':      32.8,
       'sinogramDOIcm':         0.67,
       'LORDOIcm':              0.96,
       'nRadialBins':           344,
       'nMash':                 1,
       'rCrystalDimCm':         2.0,
       'xCrystalDimCm':         0.41725,
       'zCrystalDimCm':         0.40625,
       'transaxialFovCm':       60.0,
       'span':                  11,
       'nSegments':             11,
       'maxRingDiff':           60,
       'nTofBins':              1,
       'coinciWindowWidthNsec': 5.85938,
       'tofResolutionNsec':     5.85938,
       'tofOffsetNsec':         0          
       }


import numpy as np
import os
import subprocess
from sys import platform
from scipy import ndimage
import multiprocessing as mp
np.seterr(divide='ignore')
#np.seterr(all='raise',divide='ignore') 

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

class BuildGeometry_v4:
    def __init__(self,scannerModel,radialBinCropfactor=0):

        self.scanner = dotstruct()
        self.sinogram = dotstruct()
        self.engine = dotstruct()
        self.image = dotstruct()
       
        self.sinogram.radialBinCropfactor = radialBinCropfactor
        self.fov_mask = None
        self.gaps = None
        if platform == "win32":
             self.engine.bar = '\\'
        else:
             self.engine.bar = '/'        
        
        if scannerModel.lower() == 'mct':
            self.__computeGantryInfo(mct)
        elif scannerModel.lower() == 'mmr':
            self.__computeGantryInfo(mmr)
     
    def __computeGantryInfo(self,g):
        self.__load_gantry_dict(g)
        self.scanner.nCrystalsPerBlock = self.scanner.nPhysCrystalsPerBlock + self.scanner.useVirtualCrystal
        self.scanner.nCrystalsPerRing = self.scanner.nBlockPerRing * self.scanner.nCrystalsPerBlock
        if self.scanner.model_number == 1104:
            self.scanner.nCrystalRings = self.scanner.nBlockRings * self.scanner.nPhysCrystalsPerBlock + (self.scanner.nBlockRings-1)*self.scanner.useVirtualCrystal
        elif self.scanner.model_number ==2008:
            self.scanner.nCrystalRings = self.scanner.nBlockRings * self.scanner.nPhysCrystalsPerBlock
        self.scanner.effDetectorRadiusCm = self.scanner.detectorRadiusCm + self.scanner.LORDOIcm
        self.scanner.isTof = self.sinogram.nTofBins>1
        self.scanner.TofBinWidthNsec = self.scanner.coinciWindowWidthNsec/self.sinogram.nTofBins
        self.scanner.planeSepCm = self.scanner.zCrystalDimCm/2.0        
        self.sinogram.nAngularBins = self.scanner.nCrystalsPerRing//2//self.sinogram.nMash
        self.image.matrixSize = [self.sinogram.nRadialBins,self.sinogram.nRadialBins,2*self.scanner.nCrystalRings-1]
        self.image.voxelSizeCm = [self.scanner.xCrystalDimCm/2.0,self.scanner.xCrystalDimCm/2.0,self.scanner.planeSepCm]
        self.is3d = False
        
    def __load_gantry_dict(self,g):
        self.scanner.model_number = g['model_number']
        self.scanner.circularGantry = g['circularGantry']
        self.scanner.nBuckets = g['nBuckets']
        self.scanner.nBlockRings = g['nBlockRings']
        self.scanner.nBlockPerRing = g['nBlockPerRing']
        self.scanner.nPhysCrystalsPerBlock = g['nPhysCrystalsPerBlock']
        self.scanner.useVirtualCrystal = g['useVirtualCrystal']
        self.scanner.detectorRadiusCm = g['detectorRadiusCm']
        self.scanner.sinogramDOIcm = g['sinogramDOIcm']
        self.scanner.LORDOIcm = g['LORDOIcm']
        self.scanner.rCrystalDimCm = g['rCrystalDimCm']
        self.scanner.xCrystalDimCm = g['xCrystalDimCm']
        self.scanner.zCrystalDimCm = g['zCrystalDimCm']
        self.scanner.transaxialFovCm = g['transaxialFovCm']
        self.scanner.maxRingDiff = g['maxRingDiff']
        self.scanner.coinciWindowWidthNsec = g['coinciWindowWidthNsec']
        self.scanner.tofResolutionNsec = g['tofResolutionNsec']
        self.scanner.tofOffsetNsec = g['tofOffsetNsec']    
        self.sinogram.nRadialBins_orig = g['nRadialBins']
        self.sinogram.nRadialBins = g['nRadialBins'] - int(np.ceil(g['nRadialBins']*(self.sinogram.radialBinCropfactor)/2.0)*2)
        self.sinogram.nMash = g['nMash']
        self.sinogram.span = g['span']
        self.sinogram.nSegments = g['nSegments']
        self.sinogram.nTofBins = g['nTofBins']
        
    def setTo3d(self):
        self.is3d = True
        self.fov_mask = None
         
    def setApirlMmrEngine(self,binPath=None,temPath=None,gpu=True, multiprocess = True):
        if binPath is None:
             binPath = r'C:\MatlabWorkSpace\apirl-tags\APIRL1.3.3_win64_cuda8.0_sm35\build\bin'
        if temPath is None:
             temPath = os.getcwd()
        if not os.path.exists(temPath):
             os.makedirs(temPath)
        self.engine.temPath = temPath        
        self.engine.binPath = binPath
        self.engine.gpu = gpu
        self.engine.multiprocess = multiprocess
        self.setTo3d()
        self.buildMichelogram() 
        self.sinogram.shape = [self.sinogram.nRadialBins,self.sinogram.nAngularBins,self.sinogram.totalNumberOfSinogramPlanes]
        return
        

    def buildMichelogram(self):
        a = np.transpose(np.arange(1,self.scanner.nCrystalRings**2+1).reshape(self.scanner.nCrystalRings,self.scanner.nCrystalRings))
        b = np.arange(-1*self.scanner.maxRingDiff,self.scanner.maxRingDiff + 1).reshape(self.sinogram.nSegments,self.sinogram.span)
        direction = self.sinogram.nSegments//2
        isodd = np.remainder(b[direction,0],2)
        Segments = []
        maxNumberOfPlanesPerSeg = np.zeros([self.sinogram.nSegments, 2],dtype='int16')
        
        for j in range(self.sinogram.nSegments):
            diagonalsPerSegment = []
            for i in range(self.sinogram.span):
                diagonalsPerSegment.append(np.diag(a,k=b[j,i]))
            if j == direction and isodd:
                c=0; k=1
            else:
                c=1; k=0
            oddPlanes,maxNumberOfPlanesPerSeg[j,0] = self.__zero_pad(diagonalsPerSegment[k::2]) # Odd planes
            evenPlanes,maxNumberOfPlanesPerSeg[j,1] = self.__zero_pad(diagonalsPerSegment[c::2]) # Even planes
            OddEvenPlanesPerSegment = np.empty((np.sum(maxNumberOfPlanesPerSeg[j,:]), ), dtype=object)
            OddEvenPlanesPerSegment[0::2] = self.__zero_trim(oddPlanes)
            OddEvenPlanesPerSegment[1::2] = self.__zero_trim(evenPlanes)
            Segments.append(OddEvenPlanesPerSegment)
  
        self.sinogram.numberOfPlanesPerSeg = np.sum(maxNumberOfPlanesPerSeg,axis=1)
        self.sinogram.totalNumberOfSinogramPlanes = np.sum(self.sinogram.numberOfPlanesPerSeg)
        return Segments

    def plotMichelogram(self,showRingNumber=0):
        import matplotlib.pyplot as plt
        Segments = self.buildMichelogram()
        grid = np.zeros([self.scanner.nCrystalRings**2,1],dtype='int16')
        nS = self.sinogram.nSegments
        colourPerSeg = np.concatenate([np.arange(0,(nS-1)/2), [(nS-1)/2 +1], np.arange((nS-1)/2 -1,-1,-1)]) + 1
        
        for i in range(nS):
            idx = np.concatenate(Segments[i][:])-1
            grid[idx] = colourPerSeg[i]
        grid = grid.reshape([self.scanner.nCrystalRings,self.scanner.nCrystalRings])    
        ringNumber = np.arange(1,grid.size + 1)
        plt.imshow(grid, aspect='equal')
        if showRingNumber == 1:
            k = 0
            for (j, i), _ in np.ndenumerate(grid):
                label = '{}'.format(ringNumber[k])
                plt.text(i,j,label,ha='center',va='center',fontsize=12)
                k+=1 
        ax = plt.gca();
        ax = plt.gca();
        ax.set_xticks(np.arange(0, self.scanner.nCrystalRings, 1));
        ax.set_yticks(np.arange(0, self.scanner.nCrystalRings, 1));
        ax.set_xticklabels(np.arange(1, self.scanner.nCrystalRings+1, 1));
        ax.set_yticklabels(np.arange(1, self.scanner.nCrystalRings+1, 1));
        ax.set_xticks(np.arange(-.5, self.scanner.nCrystalRings, 1), minor=True);
        ax.set_yticks(np.arange(-.5, self.scanner.nCrystalRings, 1), minor=True);
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
        plt.title('Michelogram: Span = {}, nSegments = {}'.format(self.sinogram.span,self.sinogram.nSegments),fontsize=18)
        plt.tight_layout()
        plt.show()

    def LorsAxialCoor(self):
        #Axial coordinate of LORs in each segment
        Segments = self.buildMichelogram()
        z_axis = self.scanner.zCrystalDimCm* np.arange(-(self.scanner.nCrystalRings -1)/2,(self.scanner.nCrystalRings -1)/2+1)
        axialCoorPerSeg = []
        
        for i in range(self.sinogram.nSegments):
            zy = np.zeros([self.sinogram.numberOfPlanesPerSeg[i],4])
            for j in range(self.sinogram.numberOfPlanesPerSeg[i]):
                ii,jj = self.__col2ij(Segments[i][j],self.scanner.nCrystalRings)
                zy[j,0] = np.mean(z_axis[ii])
                zy[j,1] = np.mean(z_axis[jj])
                zy[j,2] = self.scanner.effDetectorRadiusCm
                zy[j,3] = -self.scanner.effDetectorRadiusCm
            axialCoorPerSeg.append(zy)
        return axialCoorPerSeg,z_axis
    
    def plotLorsAxialCoor(self,plotSeparateSegmentsToo=0):
        import matplotlib.pyplot as plt
        axialCoorPerSeg,z_axis = self.LorsAxialCoor()
        
        plt.figure()
        for j in range(self.sinogram.nSegments):
            for i in range(len(axialCoorPerSeg[j])):
                plt.plot(axialCoorPerSeg[j][i,0:2],axialCoorPerSeg[j][i,2:4],color='green', linestyle='solid')  
        plt.plot(z_axis,self.scanner.effDetectorRadiusCm*np.ones([len(z_axis),]),'bs',fillstyle='none',markeredgewidth=2)
        plt.plot(z_axis,-self.scanner.effDetectorRadiusCm*np.ones([len(z_axis),]),'bs',fillstyle='none',markeredgewidth=2)
        plt.xlabel('Axial Distance (cm)',fontsize=18)
        plt.ylabel('Radial Distance (cm)',fontsize=18)
        plt.title('All Segments',fontsize=18)

        if plotSeparateSegmentsToo==1:
            ii = (self.sinogram.nSegments)//2
            order = np.zeros([self.sinogram.nSegments,],dtype='int16')
            order[0] = ii
            order[1::2] = np.arange(ii+1,self.sinogram.nSegments,dtype='int16')
            idx = order!=0
            order[2::2] = np.arange(ii-1,-1,-1,dtype='int16')
            q = 0       
            for j in range(self.sinogram.nSegments):
                if idx[j]:
                    if q==0:
                        seg_title = "Segment: {}"
                    else:
                        seg_title = "Segment: $\pm$ {}"
                    plt.figure()
                    plt.plot(z_axis,self.scanner.effDetectorRadiusCm*np.ones([len(z_axis),]),'bs',fillstyle='none',markeredgewidth=2)
                    plt.plot(z_axis,-self.scanner.effDetectorRadiusCm*np.ones([len(z_axis),]),'bs',fillstyle='none',markeredgewidth=2)
                    plt.xlabel('Axial Distance (cm)',fontsize=18)
                    plt.ylabel('Radial Distance (cm)',fontsize=18)
                    plt.title(seg_title.format(q),fontsize=18)
                    q+=1
                ii = order[j]
                for i in range(len(axialCoorPerSeg[ii])):
                    plt.plot(axialCoorPerSeg[ii][i,0:2],axialCoorPerSeg[ii][i,2:4],color='green', linestyle='solid')

    def LorsTransaxialCoor(self):
        #Transaxial coordinate of LORs in each segment

        startXtal = (self.scanner.nCrystalsPerRing - self.sinogram.nRadialBins)//4 #mmr = 40, mct=68 for radialBinCropfactor = 1
        self.sinogram.startXtal = startXtal
  
        p = np.linspace(2*np.pi,0,self.scanner.nCrystalsPerRing+1)
        centerCm = np.zeros([self.scanner.nCrystalsPerRing,2])
        centerCm[:,0]= self.scanner.effDetectorRadiusCm*np.cos(p[1::])
        centerCm[:,1]= self.scanner.effDetectorRadiusCm*np.sin(p[1::])
        isVirtualCrystal = np.zeros([self.scanner.nCrystalsPerRing,],dtype='bool');
        idx = np.arange(self.scanner.nPhysCrystalsPerBlock+1,self.scanner.nCrystalsPerRing+self.scanner.nPhysCrystalsPerBlock+1,
                        self.scanner.nPhysCrystalsPerBlock+1)
        isVirtualCrystal[idx-1]= 1
        
        increment = np.zeros([self.scanner.nCrystalsPerRing,2],dtype='int16')
        increment[0::2,0] = np.arange(1,self.scanner.nCrystalsPerRing/2+1)
        increment[1::2,0] = increment[0::2,0]+1
        increment[0::2,1] = np.arange(self.scanner.nCrystalsPerRing/2+1,self.scanner.nCrystalsPerRing+1)
        increment[1::2,1] = increment[0::2,1]

        halfNumberOfRadialBins = (self.sinogram.nRadialBins)//2+1 # Before interleaving
        R = np.empty((self.scanner.nCrystalsPerRing,3), dtype=object)
        V = np.zeros([halfNumberOfRadialBins,2],dtype='int16')
        
        for ii in range(self.scanner.nCrystalsPerRing):  
            s1 = (startXtal + np.arange(0, halfNumberOfRadialBins,dtype='int16')) - increment[ii,0]
            s2 = (startXtal + np.arange(0, halfNumberOfRadialBins,dtype='int16')) - increment[ii,1]
            s1 = self.__rem_p(s1, self.scanner.nCrystalsPerRing)-1
            s2 = self.__rem_p(s2, self.scanner.nCrystalsPerRing)-1
            s2 = s2[::-1]
            P1 = centerCm[s1,:]
            P2 = centerCm[s2,:]
            V = 0*V
            V[:,0] = isVirtualCrystal[s1]
            V[:,1] = isVirtualCrystal[s2]
            R[ii,0] = P1
            R[ii,1] = P2
            R[ii,2] = V    
            
        if 0: # test rotation
            import matplotlib.pyplot as plt
            plt.figure()
            for ii in range(0,self.scanner.nCrystalsPerRing,1):
                plt.clf()
                P1 = R[ii,0]
                P2 = R[ii,1]
                V = R[ii,2]
                for j in range(halfNumberOfRadialBins):
                    if V[j,0]:
                        s = 'og'
                    else:
                        s = '.b'
                    plt.plot(P1[j,0],P1[j,1],s)
                    if V[j,1]:
                        f = 'og'
                    else:
                        f = '.r'
                    plt.plot(P2[j,0],P2[j,1],f)
                    plt.axis('equal')
                    plt.xlim((-self.scanner.transaxialFovCm,self.scanner.transaxialFovCm))
                    plt.ylim((-self.scanner.transaxialFovCm,self.scanner.transaxialFovCm))
                    plt.plot([0,0],[-60,60],'--k')
                    plt.plot([-60,60],[0,0],'--k')
                    plt.plot(centerCm[:,0],centerCm[:,1],c='k',ls='--',lw=0.5)
                    plt.show()
                    plt.pause(0.1)
        # Interleaving
        xy1 = np.zeros([self.scanner.nCrystalsPerRing//2,self.sinogram.nRadialBins,2])
        xy2 = np.zeros([self.scanner.nCrystalsPerRing//2,self.sinogram.nRadialBins,2])
        gaps = np.zeros([self.scanner.nCrystalsPerRing//2,self.sinogram.nRadialBins],dtype='int16')#
        
        for i in range(self.scanner.nCrystalsPerRing//2):
            idx = self.scanner.nCrystalsPerRing-(2*i+2)
            P1=R[idx,0]
            P2=R[idx,1]
            xy1[i,0:self.sinogram.nRadialBins:2,:] = P1[0:-1,:]
            xy1[i,1:self.sinogram.nRadialBins:2,:] = (P1[0:-1,:]+P1[1::,:])/2
            xy2[i,0:self.sinogram.nRadialBins:2,:] = P2[0:-1,:]
            xy2[i,1:self.sinogram.nRadialBins:2,:] = (P2[0:-1,:]+P2[1::,:])/2
            a = np.sum(R[idx+1,2],axis=1).reshape(-1,1)>0
            b = np.sum(R[idx ,2],axis=1).reshape(-1,1)>0
            c = np.concatenate([a,b],axis=1).flatten()
            gaps[i,:]= c[1:-1]

        if self.sinogram.nMash==2:
            xy1 = (xy1[0::2,:,:] + xy1[1::2,:,:])/2
            xy2 = (xy2[0::2,:,:] + xy2[1::2,:,:])/2
            gap = np.zeros([self.sinogram.nAngularBins,self.sinogram.nRadialBins],dtype='int16')
            for i in range(self.sinogram.nAngularBins):
                gap[i,:] = np.sum(gaps[2*i:2*i+2,:],axis=0)
            gaps = gap
        gaps = np.transpose(gaps)
        
        # Calculate angular sampling
        centalBin = self.sinogram.nRadialBins//2-1
        p1 = xy1[0,centalBin,:]
        p2 = xy2[0,centalBin,:]
        lor1 = p2-p1
        p1 = xy1[1,centalBin,:]
        p2 = xy2[1,centalBin,:]
        lor2 = p2-p1
        CosTheta = np.dot(lor1,lor2)/(np.linalg.norm(lor1)*np.linalg.norm(lor2))
        self.sinogram.angSamplingDegrees = np.arccos(CosTheta)*180/np.pi                
        return xy1, xy2, gaps 

    def plotLorsTransaxialCoor(self):
        import matplotlib.pyplot as plt
        xy1, xy2, gaps = self.LorsTransaxialCoor()
        
        plt.figure()
        for i in range(self.sinogram.nAngularBins//4):
            plt.clf()
            plt.plot(np.array([xy1[i,:,0],xy2[i,:,0]]), np.array([xy1[i,:,1],xy2[i,:,1]]),c='green',ls='-',lw=0.5)
            if self.sinogram.nMash==2:
                idx = gaps[:,i]>1
            else:
                idx = gaps[:,i]>0
            
            plt.plot(np.array([xy1[i,idx,0],xy2[i,idx,0]]), np.array([xy1[i,idx,1],xy2[i,idx,1]]),c='blue',ls='-',lw=0.75)
            plt.axis('square')
            lim = self.scanner.transaxialFovCm*3/4
            plt.xlim((-lim,lim))
            plt.ylim((-lim,lim))
            plt.plot([0,0],[-lim,lim],'--k')
            plt.plot([-lim,lim],[0,0],'--k')
            plt.title('Angle: {}'.format(str(i+1)),fontsize=15)
            plt.show()
            plt.pause(0.1)  

    def Lors3DEndPointCoor(self,reduce4symmetries = 0):
        axialCoorPerSeg,_= self.LorsAxialCoor()
        xy1, xy2, gaps = self.LorsTransaxialCoor()
        xyz01 = np.zeros([self.sinogram.nAngularBins,self.sinogram.nRadialBins,3,self.sinogram.totalNumberOfSinogramPlanes],dtype='float32')
        xyz02 = np.zeros([self.sinogram.nAngularBins,self.sinogram.nRadialBins,3,self.sinogram.totalNumberOfSinogramPlanes],dtype='float32')
        k = 0
        flag = 1
        centralSegment = (self.sinogram.nSegments)//2
        for j in range(self.sinogram.nSegments):
            for i in range(self.sinogram.numberOfPlanesPerSeg[j]):
                z1 = axialCoorPerSeg[j][i,0]*np.ones([self.sinogram.nAngularBins,self.sinogram.nRadialBins])
                z2 = axialCoorPerSeg[j][i,1]*np.ones([self.sinogram.nAngularBins,self.sinogram.nRadialBins])
                if j>centralSegment: # for angles greater than 180
                    tmp = z1
                    z1 = z2
                    z2 = tmp
                    if flag:
                        xy1 = -xy1
                        xy2 = -xy2
                        flag = 0
                xyz01[:,:,0:2,k] = xy1
                xyz01[:,:,2,k] = z1
                xyz02[:,:,0:2,k] = xy2
                xyz02[:,:,2,k] = z2
                k+=1
        # sort the coordinate for segments 0 +1 -1 +2 -2,...
        cumulativePlaneNumber = np.cumsum(self.sinogram.numberOfPlanesPerSeg)
        planeRange = np.zeros([len(cumulativePlaneNumber),2],dtype='int16')
        planeRange[0,0] = 0
        planeRange[1:,0] = cumulativePlaneNumber[0:-1]
        planeRange[:,1] = cumulativePlaneNumber
        
        o = np.zeros([self.sinogram.nSegments,],dtype='int16')
        o[0::2] = np.arange(centralSegment,self.sinogram.nSegments)
        o[1::2] = np.arange(centralSegment-1,-1,-1)
        
        newCumulativePlaneNumber = np.cumsum(self.sinogram.numberOfPlanesPerSeg[o])
        newPlaneRange = np.zeros([len(newCumulativePlaneNumber),2],dtype='int16')
        newPlaneRange[0,0] = 0
        newPlaneRange[1:,0] = newCumulativePlaneNumber[0:-1]
        newPlaneRange[:,1] = newCumulativePlaneNumber
        
        self.sinogram.numberOfPlanesPerSeg = self.sinogram.numberOfPlanesPerSeg[o]
        self.sinogram.originalSegmentOrder = o
        
        if self.scanner.model_number ==2008:
            S = 0*newPlaneRange
            S[0,:] = newPlaneRange[0,:]
            for i in range(centralSegment):
                S[2*i+1,:] = newPlaneRange[2*i+2,:]
                S[2*i+2,:] = newPlaneRange[2*i+1,:]                      
            newPlaneRange = S
        self.sinogram.planeRange = newPlaneRange
        xyz1 = 0*xyz01
        xyz2 = 0*xyz02
        for i in range(self.sinogram.nSegments):
            xyz1[:,:,:,newPlaneRange[i,0]:newPlaneRange[i,1]]= xyz01[:,:,:,planeRange[o[i],0]:planeRange[o[i],1]]
            xyz2[:,:,:,newPlaneRange[i,0]:newPlaneRange[i,1]]= xyz02[:,:,:,planeRange[o[i],0]:planeRange[o[i],1]]
        if reduce4symmetries==1:
            self.calculateAxialSymmetries()
            xyz1 = xyz1[0:self.sinogram.nAngularBins//2,:,:,self.sinogram.uniqueAxialPlanes-1]
            xyz2 = xyz2[0:self.sinogram.nAngularBins//2,:,:,self.sinogram.uniqueAxialPlanes-1]

        return xyz1, xyz2, newPlaneRange

    def plotLors3DEndPointCoor(self,planeNumber=151):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        
        xyz1, xyz2, _ = self.Lors3DEndPointCoor()
        p = np.linspace(2*np.pi,0,self.scanner.nCrystalsPerRing+1)
        centerCm = np.zeros([self.scanner.nCrystalsPerRing,2])
        centerCm[:,0]= self.scanner.effDetectorRadiusCm*np.cos(p[1::])
        centerCm[:,1]= self.scanner.effDetectorRadiusCm*np.sin(p[1::])
                
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        i = 0
        x1 = xyz1[i,:,0,planeNumber]
        y1 = xyz1[i,:,1,planeNumber]
        z1 = xyz1[i,:,2,planeNumber]
        x2 = xyz2[i,:,0,planeNumber]
        y2 = xyz2[i,:,1,planeNumber]
        z2 = xyz2[i,:,2,planeNumber]

        for j in range(0,self.sinogram.nRadialBins,3):
            ax.plot(np.array([x1[j],x2[j]]),np.array([y1[j],y2[j]]),np.array([z1[j],z2[j]]),c='green',ls='-',lw=0.75)
        ax.plot(centerCm[0::3,0],centerCm[0::3,1],zs = z1[0],c='blue', markersize = 5, marker = 's', ls='None')
        ax.plot(centerCm[0::3,0],centerCm[0::3,1],zs = z2[0],c='blue', markersize = 5, marker = 's', ls='None')
        plt.show()        

    def calculateAxialSymmetries(self):                   
        newPlaneRange = self.sinogram.planeRange
        newPlaneRange[:,0] +=1                  
        l = self.sinogram.span//2 + 1   
        c = self.sinogram.nSegments//2           
        K = np.zeros([c,l],dtype='int16')
        K[0:c,0] = newPlaneRange[1::2,0] 
        K[0:c,1] = K[0:c,0] + 1   
        for i in range(2,l):
            K[0:,i] = K[0:,i-1]+2             
        self.sinogram.uniqueAxialPlanes = np.concatenate([[1],K.flatten()])

        ## calculate the translational and mirror symmetries
        b = newPlaneRange.flatten()
        b = np.reshape(b[2::],(4,c),order='F').transpose()
        n = self.sinogram.span-1 
        I = np.zeros([n,4],dtype='int16')
        x = np.arange(n)
        P = []
        for i in range(c):
            a = b[i,:]
            I = 0*I
            I[:,0] = a[0] + x
            I[:,1] = a[1] - x
            I[:,2] = a[2] + x
            I[:,3] = a[3] - x
            P.append(I)
        P = np.concatenate(P[:],axis=0)
        
        symID = np.zeros([P.shape[0],],dtype='int16')
        for i in range(1,len(self.sinogram.uniqueAxialPlanes)):
            j = P[:,0] == self.sinogram.uniqueAxialPlanes[i]
            symID[j] = i+1
        i = np.array(np.nonzero(symID==0),dtype='int16')
        symID[i] = symID[i-1]
        
        Ax = np.zeros([self.sinogram.totalNumberOfSinogramPlanes,],dtype='int16')
        mirror = np.ones([self.sinogram.totalNumberOfSinogramPlanes,],dtype='int16')
        Ax[0:self.sinogram.numberOfPlanesPerSeg[0]] = 1
        idx = n*np.arange(1,c+1)
        
        for i in range(len(symID)):
            if np.any((i+1)==idx):
                ii = np.concatenate([np.arange(P[i,0],P[i,1]+1),np.arange(P[i,2],P[i,3]+1)])
                Ax[ii-1] = symID[i]
                ii = np.arange(P[i,2],P[i,3])
                mirror[ii-1] = -1
            Ax[P[i,:]-1] = symID[i]
            mirror[P[i,2:4]-1] = -1

        offset = np.zeros([self.sinogram.totalNumberOfSinogramPlanes,],dtype='int16')
        for i in range(self.sinogram.totalNumberOfSinogramPlanes):
            if mirror[i]==1:
                offset[i] = (i+1) - self.sinogram.uniqueAxialPlanes[Ax[i]-1]
            else:
                j = np.array(np.nonzero(symID==(Ax[i]))[0])[0]
                x = P[j,0:3]
                offset[i] = (self.image.matrixSize[2]-1) -(x[1]-x[0]) + ((i+1) - x[2])
        planeMirrorTranslation = np.zeros([self.sinogram.totalNumberOfSinogramPlanes,3],dtype='int16')
        planeMirrorTranslation[:,0] = Ax
        planeMirrorTranslation[:,1] = mirror
        planeMirrorTranslation[:,2] = offset
        self.sinogram.planeMirrorTranslation = planeMirrorTranslation

              
    def calculateSystemMatrixPerPlane(self,xyz1,xyz2,I,reconFovRadious=None):
        if reconFovRadious is None:
            reconFovRadious = self.scanner.transaxialFovCm/2.5
        def paramInterSectPoint(p1x,p2x, amin,axmin,amax,axmax, tx,bx,dx,Nx):
            if  p1x < p2x:
                if amin == axmin:
                    imin = 1
                else:
                    imin = np.ceil(((p1x + amin*tx) - bx)/dx)
                if amax == axmax:
                    imax = Nx-1
                else:
                    imax = np.floor(((p1x + amax*tx) - bx)/dx)
                ax = (bx + np.arange(imin,imax+1)*dx - p1x ) / tx
            else:
                if amin == axmin:
                    imax = Nx-2
                else:
                    imax = np.floor(((p1x + amin*tx) - bx)/dx)
                if amax == axmax:
                    imin = 0
                else:
                    imin = np.ceil(((p1x + amax*tx) - bx)/dx)
                ax = (bx + np.arange(imax,imin-1,-1)*dx - p1x ) / tx
            return ax
        # Start Siddon tracing
        Nx = self.image.matrixSize[0] + 1
        Ny = self.image.matrixSize[1] + 1
        Nz = self.image.matrixSize[2] + 1
        dx = self.image.voxelSizeCm[0]
        dy = self.image.voxelSizeCm[1]
        dz = self.image.voxelSizeCm[2]
        bx = -(Nx-1)*dx/2
        by = -(Ny-1)*dy/2
        bz = -(Nz-1)*dz/2
        vCenter_y = dx*np.arange(-(Nx-2)/2,(Nx-2)/2+1)
        vCenter_x = dy*np.arange(-(Ny-2)/2,(Ny-2)/2+1)
        vCenter_z = dz*np.arange(-(Nz-2)/2,(Nz-2)/2+1)
        thresholdOfWeakLors = 50
        sMatrix = np.zeros((self.sinogram.nAngularBins//2,self.sinogram.nRadialBins), dtype=object)
        if self.scanner.isTof:
            from scipy.stats import norm
            tofBinBoundaries = np.linspace(-self.scanner.coinciWindowWidthNsec/2,self.scanner.coinciWindowWidthNsec/2,self.sinogram.nTofBins+1)
            sigma = self.scanner.tofResolutionNsec/np.sqrt(np.log(256))
            tofMatrix = np.zeros((self.sinogram.nAngularBins//2,self.sinogram.nRadialBins), dtype=object)
    
        for ang in range(self.sinogram.nAngularBins//2):
            for rad in range(self.sinogram.nRadialBins): 
    
                p1x = xyz1[ang,rad,0,I]
                p1y = xyz1[ang,rad,1,I]
                p1z = xyz1[ang,rad,2,I]
                p2x = xyz2[ang,rad,0,I]
                p2y = xyz2[ang,rad,1,I]
                p2z = xyz2[ang,rad,2,I]
                tx = p2x - p1x
                if tx == 0: 
                    p2x += 1e-2
                    tx = p2x - p1x
                ax = (bx + np.array([0 , Nx-1])*dx - p1x)/ tx
                axmin = ax.min()
                axmax = ax.max()
                ty = p2y - p1y
                if ty == 0: 
                    p2y += 1e-2
                    ty = p2y - p1y
                ay = (by + np.array([0 , Ny-1])*dy - p1y)/ ty
                aymin = ay.min()
                aymax = ay.max()
                tz = p2z - p1z
                if tz == 0: 
                    p1z += 1e-2
                    tz = p2z - p1z
                az = (bz + np.array([0 , Nz-1])*dz - p1z)/ tz
                azmin = az.min()
                azmax = az.max()
    
                amin = np.array([0,axmin, aymin, azmin]).max()
                amax = np.array([1,axmax, aymax, azmax]).min()
                
                if amin < amax:
                    ax = paramInterSectPoint(p1x,p2x, amin,axmin,amax,axmax, tx,bx,dx,Nx)
                    ay = paramInterSectPoint(p1y,p2y, amin,aymin,amax,aymax, ty,by,dy,Ny)
                    az = paramInterSectPoint(p1z,p2z, amin,azmin,amax,azmax, tz,bz,dz,Nz)
                    a = np.unique(np.concatenate(([[amin],ax,ay,az,[amax]])))
                    k = np.arange(len(a)-1)
                    im = np.floor(((p1x + ((a[k+1] + a[k])/2)*tx) - bx)/dx)
                    jm = np.floor(((p1y + ((a[k+1] + a[k])/2)*ty) - by)/dy)
                    km = np.floor(((p1z + ((a[k+1] + a[k])/2)*tz) - bz)/dz)
                    LorIntersectionLength = (a[k+1]-a[k])*np.sqrt(tx**2 + ty**2 + tz**2)*1e4/dx #normaized by pixel size
                    
                    M = np.stack([im, jm, km, LorIntersectionLength]).transpose().astype('int16')
                    #remove weak LOR interactions
                    weaksID = M[:,3]>thresholdOfWeakLors
                    M = M[weaksID,:]
                    if reconFovRadious!=0:# remove LOR interactions out of reduced reconstruction FOV 
                        IdsToKeep = (vCenter_y[M[:,0]]**2 + vCenter_x[M[:,1]]**2)<(reconFovRadious)**2
                        M = M[IdsToKeep,:]

                    if M.size!=0:
                        sMatrix[ang,rad] = M
                        if self.scanner.isTof:
                            VoxelCenters = np.stack([vCenter_y[M[:,0]],vCenter_x[M[:,1]],vCenter_z[M[:,2]]]).transpose()
                            nEmiPoint = VoxelCenters.shape[0]
                            endPoint1 = np.tile(np.array([p1x, p1y, p1z]),(nEmiPoint,1))
                            endPoint2 = np.tile(np.array([p2x, p2y, p2z]),(nEmiPoint,1))
                            dL = np.sqrt(np.sum((endPoint2 - VoxelCenters)**2 ,axis=1)) - np.sqrt(np.sum((endPoint1 - VoxelCenters)**2 ,axis=1))
                            dT = dL/30 - self.scanner.tofOffsetNsec
                            tofWeights = np.zeros([nEmiPoint,self.sinogram.nTofBins])
                            
                            for q in range(nEmiPoint):
                                tmp = norm.cdf(tofBinBoundaries,dT[q],sigma)
                                tofWeights[q,:] = tmp[1:] - tmp[:-1]
                            tofMatrix[ang,rad] = (tofWeights*1e4).astype('int16')
                        
        if not self.scanner.isTof:
            tofMatrix = 0
        return sMatrix,tofMatrix    
    
    def buildSystemMatrixUsingSymmetries(self,save_dir=None, reconFovRadious=None, is3d = False,ncores = 1):
        import os
        if save_dir is None:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("save path: {}".format(save_dir))
        save_dir = save_dir + self.engine.bar
        if reconFovRadious is None:
            reconFovRadious = self.scanner.transaxialFovCm/2.5
        parameters = {'reconFovRadious': reconFovRadious, 'radialBinCropfactor': self.sinogram.radialBinCropfactor}
        np.save(save_dir+'parameters.npy', parameters)
        xyz1, xyz2, _ = self.Lors3DEndPointCoor(1)
        N = len(self.sinogram.uniqueAxialPlanes) if is3d else 1
        if ncores==1:
            for i in range(N):
                print(i)
                geoMatrix,tofMatrix = self.calculateSystemMatrixPerPlane(xyz1,xyz2,i,reconFovRadious)
                np.save(save_dir+'geoMatrix-'+str(i)+'.npy', geoMatrix)
                if self.scanner.isTof:
                    np.save(save_dir+'tofMatrix-'+str(i)+'.npy', tofMatrix)
        else:
            print('to do: multiprocessing')
    
    def loadSystemMatrix(self,save_dir,is3d = False, tof=True,reconFovRadious=None):
        import time
        """
        if case usage of: 
            PET = BuildGeometry('mct') 
            PET.loadSystemMatrix(save_dir)
            call "self.Lors3DEndPointCoor(1)" internally to set sinogram attributes 
        """
        self.is3d = is3d
        save_dir = save_dir + self.engine.bar
        tic = time.time()
        if (not is3d and not os.path.isfile(save_dir+'geoMatrix-0.npy')) or (is3d and not os.path.isfile(save_dir+'geoMatrix-1.npy')):
             self.buildSystemMatrixUsingSymmetries(save_dir,is3d = is3d, reconFovRadious=reconFovRadious)
                  
        self.geoMatrix = []
        param = np.load(save_dir+'parameters.npy',allow_pickle=True).item()
        self.image.reconFovRadious = param['reconFovRadious']
        if self.is3d:
             if not hasattr(self.sinogram,'uniqueAxialPlanes'):
                 self.Lors3DEndPointCoor(1);
             if tof and self.scanner.isTof:
                 self.tofMatrix = []
             N = len(self.sinogram.uniqueAxialPlanes)
             for i in range(N):
                 self.geoMatrix.append(np.load(save_dir+'geoMatrix-'+str(i)+'.npy'))
                 if tof and self.scanner.isTof:
                     self.tofMatrix.append(np.load(save_dir+'tofMatrix-'+str(i)+'.npy'))
        else:
             self.buildMichelogram()
             if self.sinogram.radialBinCropfactor !=param['radialBinCropfactor']:
                  raise ValueError(f"Current radialBinCropfactor does't match that '{save_dir+'geoMatrix-0.npy'}', choose a different path to recompute system matrix")
             self.geoMatrix.append(np.load(save_dir+'geoMatrix-0.npy',allow_pickle=True))
             if tof and self.scanner.isTof:
                  self.tofMatrix = []
                  self.tofMatrix.append(np.load(save_dir+'tofMatrix-0.npy',allow_pickle=True))
             
        print('loaded in: {} sec.'.format(time.time()-tic))
    
        
    ''' PRIVATE HELPERS '''          
    def __zero_pad(self,y):
        maxNumberOfPlanes = [len(y[i]) for i in range(len(y))]
        PlaneNumbers = np.zeros([len(y),np.max(maxNumberOfPlanes)],dtype='int16')
        for i in range(len(y)):
            ii = (np.max(maxNumberOfPlanes) - len(y[i]))//2
            if ii==0:
                PlaneNumbers[i,:] = y[i]
            else:
                PlaneNumbers[i,ii:-ii] = y[i]
        return PlaneNumbers, np.max(maxNumberOfPlanes)
    
    def __zero_trim(self,y):
        out = []
        for i in range(len(y[0])):
            tmp = y[:,i]
            out.append(tmp[np.nonzero(tmp)])
        return out  

    def __col2ij(self,m,n):
        if np.max(m) > n**2:
            raise ValueError("m is greater than the max number of elements") 
        j = np.ceil(m/n)-1
        i = m-j*n-1
        return i.astype(int),j.astype(int) 
    
    def __rem_p(self,x,nx):
        for i in range(len(x)):
            while x[i] < nx:
                x[i] += nx
            while x[i] > nx:
                x[i] -= nx
        return x

    def gaussFilter(self,img,fwhm,is3d=False,batch_size=1):
        # 3D aniso/isotropic Gaussian filtering
        
        fwhm = np.array(fwhm)
        if np.all(fwhm==0):
            return img
        from scipy import ndimage
        if is3d:
            img = img.reshape(self.image.matrixSize,order='F')
            voxelSizeCm = self.image.voxelSizeCm
        else:
            img = img.reshape(self.image.matrixSize[:2],order='F')
            voxelSizeCm = self.image.voxelSizeCm[:2]
        if fwhm.shape==1:
            if is3d:
                fwhm=fwhm*np.ones([3,])
            else:
                fwhm=fwhm*np.ones([2,])
        sigma=fwhm/voxelSizeCm/np.sqrt(2**3*np.log(2))
        imOut = ndimage.filters.gaussian_filter(img,sigma)
        return imOut.flatten('F')        
        
    def buildPhantom(self,model = 0,display = False):
        if model==0: # shepp-logan like phantom
            x = np.arange(-self.image.matrixSize[0]//2, self.image.matrixSize[0]//2, 1)
            y = np.arange(-self.image.matrixSize[1]//2, self.image.matrixSize[1]//2, 1)
            z = np.arange(-self.image.matrixSize[2]//2, self.image.matrixSize[2]//2, 1)
            xx, yy,zz = np.meshgrid(x, y,z)
            
            z1 = (xx**2+(yy/1.3)**2+((zz)/0.8)**2)<80**2
            z2 = 2.5*(((xx+20)**2+(yy+20)**2+(zz/3)**2)<10**2)
            z3 = 3*(((xx/0.45)-60)**2+(yy+20)**2+(zz/3)**2)<60**2
            z4 = 2*(((xx)**2+(yy/0.8-30)**2+(zz/3)**2)<15**2)
            img = z1.astype('float') + z2.astype('float')+ z3.astype('float')+ z4.astype('float')
        else:
            raise ValueError("unknown phantom")
        
        if display:
            import matplotlib.pyplot as plt
            slices = np.sort(np.random.randint(self.image.matrixSize[2],size=4))
            plt.figure()
            plt.subplot(2,2,1),plt.imshow(img[:,:,slices[0]]),plt.title('Slice: {}'.format(str(slices[0])),fontsize=15)
            plt.subplot(2,2,2),plt.imshow(img[:,:,slices[1]]),plt.title('Slice: {}'.format(str(slices[1])),fontsize=15)
            plt.subplot(2,2,3),plt.imshow(img[:,:,slices[2]]),plt.title('Slice: {}'.format(str(slices[2])),fontsize=15)
            plt.subplot(2,2,4),plt.imshow(img[:,:,slices[3]]),plt.title('Slice: {}'.format(str(slices[3])),fontsize=15)
            plt.show()
        return img
    
    def bit_reverse(self,mm):
        
        dec2bin = lambda x,y: [np.binary_repr(x[i], width=y) for i in range(len(x))]
        bin2dec_reverse = lambda x: np.array([int(x[i][::-1],2) for i in range(len(x))])
        nn = 2**np.ceil(np.log2(mm)).astype(int) 
        y = len(np.binary_repr(nn-1))
        ii = bin2dec_reverse(dec2bin(np.arange(nn),y))  
        ii = ii[ii < mm]
        return ii
    
    def check_nsubs(self,nsub):
         nAngles = self.sinogram.nAngularBins
         if (nAngles % nsub) != 0:
              i = np.arange(1,nAngles)
              j = (nAngles % i)== 0
              raise ValueError(f'Choose a valid subset: {i[j]}')
          
    def angular_subsets(self,nsub):
        nAngles = self.sinogram.nAngularBins
        if (nAngles//nsub) % 2 != 0:
            i = np.arange(1,nAngles/2)
            j = (np.mod(nAngles/2/i,1))== 0
            raise ValueError(f'Choose a valid subset: {i[j]}')
    
        subsize = int(nAngles/nsub)
        subsets = np.zeros((subsize, nsub),dtype='int16')
        for j in range(nsub):
            k = 0
            x = np.arange(j,nAngles//2,nsub)
            for i in x:
                subsets[k,j] = i
                subsets[k+subsize//2,j] = i+nAngles/2
                k+=1
        s = subsets
        subsets = 0*subsets
        st = self.bit_reverse(nsub)
        for i in range(nsub):
            subsets[:,i] = s[:,st[i]]
    
        return subsets, subsize
    

    def forwardProject3D(self,img3d,tof=False, psf=0):
        import time
        if tof and not self.scanner.isTof:
           raise ValueError("The scanner is not TOF") 
        nUniqueAxialPlanes = len(self.sinogram.uniqueAxialPlanes)
        allPlanes = []
        for i in range(len(self.sinogram.uniqueAxialPlanes)):
            allPlanes.append(np.nonzero(self.sinogram.planeMirrorTranslation[:,0] == i+1)[0])
            
        img3d = self.gaussFilter(img3d.flatten('F'),psf)
        dims = [self.sinogram.nRadialBins,self.sinogram.nAngularBins,self.sinogram.totalNumberOfSinogramPlanes]
        if tof: dims.append(self.sinogram.nTofBins)
        y = np.zeros(dims,dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2
        planeMirrorTranslation = self.sinogram.planeMirrorTranslation
        prod = lambda x, y: x.reshape(-1,1).dot(y.reshape(1,-1)).T.astype('int32')
        tic = time.time()
        for i in range(self.sinogram.nAngularBins//2):
            for j in range(self.sinogram.nRadialBins):
                for p in range(nUniqueAxialPlanes):
                    M0 = self.geoMatrix[p][i,j]
                    if not np.isscalar(M0):
                        M = M0[:,0:3].astype('int32')
                        G = M0[:,3]/1e4
                        H = planeMirrorTranslation[allPlanes[p],:]
                        idxAxial = matrixSize[0]*matrixSize[1]*(prod(H[:,1],M[:,2]) + H[:,2])
                        idx1 = (M[:,0] + M[:,1]*matrixSize[0]).reshape(-1,1) + idxAxial
                        idx2 = (M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])).reshape(-1,1) + idxAxial
#                        if tof:
#                            W = self.tofMatrix[p][i,j]/1e4
#                            y[j,i,allPlanes[p],:] = (G.reshape(-1,1)*img3d[idx1]).T.dot(W)
#                            y[j,i+q,allPlanes[p],:] = (G.reshape(-1,1)*img3d[idx2]).T.dot(W)
#                        else:
                        y[j,i,allPlanes[p]] = G.dot(img3d[idx1])
                        y[j,i+q,allPlanes[p]] = G.dot(img3d[idx2]) 
        print('forward-projected in: {} sec.'.format(time.time()-tic))                    
        return y 

    def MLEM3D_python(self, prompts,img=None,RS=None,niter=100, AN=None, tof=False, psf=0):
        import time
        if tof and not self.scanner.isTof:
               raise ValueError("The scanner is not TOF") 
        if img is None:
            img = np.ones(self.image.matrixSize,dtype='float')
        img = img.flatten('F')  
        sensImage = np.zeros_like(img)
        if np.ndim(prompts)!=4:
            tof = False
        if RS is None:
            RS = 0*prompts
        if AN is None:
            AN = np.ones([self.sinogram.nRadialBins,self.sinogram.nAngularBins,self.sinogram.totalNumberOfSinogramPlanes],dtype='float')
    
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2
        Flag = True    
    
        nUniqueAxialPlanes = len(self.sinogram.uniqueAxialPlanes)
        allPlanes = []
        for i in range(len(self.sinogram.uniqueAxialPlanes)):
            allPlanes.append(np.nonzero(self.sinogram.planeMirrorTranslation[:,0] == i+1)[0])
        planeMirrorTranslation = self.sinogram.planeMirrorTranslation
        prod = lambda x, y: x.reshape(-1,1).dot(y.reshape(1,-1)).T.astype('int32')
        
        tic = time.time()
        for n in range(niter):    
            if np.any(psf!=0):
                imgOld = self.gaussFilter(img,psf,True)
            else:
                imgOld = img
            backProjImage = 0*img 
            for i in range(self.sinogram.nAngularBins//2):
                for j in range(self.sinogram.nRadialBins):
                    for p in range(nUniqueAxialPlanes):
                        M0 = self.geoMatrix[p][i,j]
                        if not np.isscalar(M0):
                            M = M0[:,0:3].astype('int32')
                            G = M0[:,3]/1e4
                            H = planeMirrorTranslation[allPlanes[p],:]
                            idxAxial = matrixSize[0]*matrixSize[1]*(prod(H[:,1],M[:,2]) + H[:,2])
                            idx1 = (M[:,0] + M[:,1]*matrixSize[0]).reshape(-1,1) + idxAxial
                            idx2 = (M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])).reshape(-1,1) + idxAxial
                            if tof:
                                W = self.tofMatrix[p][i,j]/1e4
                                f1 = AN[j,i,allPlanes[p]].reshape(-1,1)*(prompts[j,i,allPlanes[p],:]/ ((G.reshape(-1,1)*imgOld[idx1]).T.dot(W) + RS[j,i,allPlanes[p],:]+1e-5))
                                f2 = AN[j,i+q,allPlanes[p]].reshape(-1,1)*(prompts[j,i+q,allPlanes[p],:]/ ((G.reshape(-1,1)*imgOld[idx2]).T.dot(W) + RS[j,i+q,allPlanes[p],:]+1e-5))                           
                                backProjImage[idx1] += G.reshape(-1,1)*W.dot(f1.T)
                                backProjImage[idx2] += G.reshape(-1,1)*W.dot(f2.T)
                            else:
                                f1 = AN[j,i,allPlanes[p]]*(prompts[j,i,allPlanes[p]]/(AN[j,i,allPlanes[p]]*G.dot(imgOld[idx1])+RS[j,i,allPlanes[p]]+1e-5))
                                f2 = AN[j,i+q,allPlanes[p]]*(prompts[j,i+q,allPlanes[p]]/(AN[j,i+q,allPlanes[p]]*G.dot(imgOld[idx2])+RS[j,i+q,allPlanes[p]]+1e-5))
                                backProjImage[idx1] += G.reshape(-1,1).dot(f1.reshape(1,-1))
                                backProjImage[idx2] += G.reshape(-1,1).dot(f2.reshape(1,-1)) 
                            if Flag:
                                if tof:
                                    GW = G*np.sum(W,axis = 1)
                                    sensImage[idx1] += GW.reshape(-1,1).dot(AN[j,i,allPlanes[p]].reshape(1,-1)) 
                                    sensImage[idx2] += GW.reshape(-1,1).dot(AN[j,i+q,allPlanes[p]].reshape(1,-1)) 
                                else:
                                    sensImage[idx1] += G.reshape(-1,1).dot(AN[j,i,allPlanes[p]].reshape(1,-1)) 
                                    sensImage[idx2] += G.reshape(-1,1).dot(AN[j,i+q,allPlanes[p]].reshape(1,-1)) 
            if np.any(psf!=0) and Flag:
                sensImage = self.gaussFilter(sensImage,psf,True)
            Flag = False
            img = imgOld*backProjImage/(sensImage+1e-5)
                          
        print('forward-projected in: {} sec.'.format((time.time()-tic)/60))                    
        return img.reshape(matrixSize,order='F') 
   
    def backProjectBatch2D(self, sinodata=None, tof=False, psf=0):

        if tof and not self.scanner.isTof:
               raise ValueError("The scanner is not TOF") 
        if sinodata is None:
            batch_size = 1
            if tof:
                sinodata = np.ones([batch_size, self.sinogram.nRadialBins,self.sinogram.nAngularBins,self.sinogram.nTofBins],dtype='float')
            else:
                sinodata = np.ones([batch_size, self.sinogram.nRadialBins,self.sinogram.nAngularBins],dtype='float')  
        else:
            if (tof and np.ndim(sinodata)==4) or (not tof and np.ndim(sinodata)==3):
                 batch_size = sinodata.shape[0] 
            else:
                 batch_size = 1
                 sinodata=sinodata[None]

        img = np.zeros([batch_size,np.prod(self.image.matrixSize[:2])],dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2
    
        for i in range(self.sinogram.nAngularBins//2):
            for j in range(self.sinogram.nRadialBins):
                M0 = self.geoMatrix[0][i,j]
                if not np.isscalar(M0):
                    M = M0[:,0:3].astype('int32')
                    G = M0[:,3]/1e4
                    idx1 = M[:,0] + M[:,1]*matrixSize[0]
                    idx2 = M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])
                    if tof:
                        W = self.tofMatrix[0][i,j]/1e4
                        for b in range(batch_size):
                            img[b,idx1] += G*W.dot(sinodata[b,j,i,:])
                            img[b,idx2] += G*W.dot(sinodata[b,j,i+q,:])
                    else:
                        for b in range(batch_size):
                            img[b,idx1] += G*sinodata[b,j,i]
                            img[b,idx2] += G*sinodata[b,j,i+q]
        if np.any(psf!=0):
            for b in range(batch_size):
                img[b,:] = self.gaussFilter(img[b,:],psf)
        img = np.reshape(img,[batch_size,matrixSize[0],matrixSize[1]],order='F')
        if batch_size ==1:
             img = img[0,:,:]
        return img

    def forwardProjectBatch2D(self,img, tof=False, psf=0):
        if tof and not self.scanner.isTof:
           raise ValueError("The scanner is not TOF")    
        if np.ndim(img)==2:
            batch_size = 1
            img = img[None,:,:]
        else:
            batch_size = img.shape[0]
        img = img.reshape([batch_size,np.prod(img.shape[1:3])],order='F')
        if psf:
            for b in range(batch_size):
                img[b,:] = self.gaussFilter(img[b,:],psf)
        dims = [batch_size, self.sinogram.nRadialBins,self.sinogram.nAngularBins]
        if tof: dims.append(self.sinogram.nTofBins)
        y = np.zeros(dims,dtype='float32')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2

        for i in range(self.sinogram.nAngularBins//2):
            for j in range(self.sinogram.nRadialBins):
                M0 = self.geoMatrix[0][i,j]
                if not np.isscalar(M0):
                    M = M0[:,0:3].astype('int32')
                    G = M0[:,3]/1e4
                    idx1 = M[:,0] + M[:,1]*matrixSize[0]
                    idx2 = M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])
                    if tof:
                        W = self.tofMatrix[0][i,j]/1e4
                        for b in range(batch_size):
                            y[b,j,i,:] = (G*img[b,idx1]).dot(W)
                            y[b,j,i+q,:] =(G*img[b,idx2]).dot(W)
                    else:
                        for b in range(batch_size):
                            y[b,j,i] = G.dot(img[b,idx1])
                            y[b,j,i+q] = G.dot(img[b,idx2]) 
        if batch_size==1:
            if tof:
                y = y[0,:,:,:]
            else:
                y = y[0,:,:]
        print(f'{batch_size} batches forward-projected\n')                    
        return y

    def OSMAPEM2D_DePierro(self, prompts,img=None,RS=None,niter=100, nsubs=1, AN=None, tof=False, psf=0, beta=1, prior = None, prior_weights = 1):
        import time
        [numAng,subSize] = self.angular_subsets(nsubs)
        if tof and not self.scanner.isTof:
               raise ValueError("The scanner is not TOF") 
        if img is None:
            img = np.ones(self.image.matrixSize[:2],dtype='float')
        img = img.flatten('F')
        sensImage = np.zeros_like(img)
        sensImageSubs = np.zeros((np.prod(self.image.matrixSize[:2]),nsubs),dtype='float')
        if np.ndim(prompts)!=3:
            tof = False
        if RS is None:
            RS = 0*prompts
        if AN is None:
            AN = np.ones([self.sinogram.nRadialBins,self.sinogram.nAngularBins],dtype='float')
        matrixSize = self.image.matrixSize
        if prior is None:
            from geometry.Prior import Prior
            prior = Prior(matrixSize[:2])
        
        W = prior.Wd*prior_weights   
        wj = prior.imCropUndo(W.sum(axis=1))
        
        display = 0
        if display:
            import matplotlib.pyplot as plt
            plt.figure(), 
        
        q = self.sinogram.nAngularBins//2
        Flag = True
        tic = time.time()
        for n in range(niter):
            for sub in range(nsubs):
                if np.any(psf!=0):
                    imgOld = self.gaussFilter(img,psf)
                else:
                    imgOld = img
                backProjImage = 0*img 
                sensImage = 0*sensImage
                for ii in range(subSize//2):
                    i = numAng[ii,sub]
                    for j in range(self.sinogram.nRadialBins):
                        M0 = self.geoMatrix[0][i,j]
                        if not np.isscalar(M0):
                            M = M0[:,0:3].astype('int32')
                            G = M0[:,3]/1e4
                            idx1 = M[:,0] + M[:,1]*matrixSize[0]
                            idx2 = M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])
                            if tof:
                                W = self.tofMatrix[0][i,j]/1e4
                                backProjImage[idx1] += G*AN[j,i]*W.dot(prompts[j,i,:]/(AN[j,i]*(G*img[idx1]).dot(W)+RS[j,i,:]+1e-5))
                                backProjImage[idx2] += G*AN[j,i+q]*W.dot(prompts[j,i+q,:]/(AN[j,i+q]*(G*img[idx2]).dot(W)+RS[j,i+q,:]+1e-5))
                            else:
                                backProjImage[idx1] += G*AN[j,i]*(prompts[j,i]/(AN[j,i]*G.dot(imgOld[idx1])+RS[j,i]+1e-5))
                                backProjImage[idx2] += G*AN[j,i+q]*(prompts[j,i+q]/(AN[j,i+q]*G.dot(imgOld[idx2])+RS[j,i+q]+1e-5)) 
                            if Flag:
                                if tof:
                                    GW = G*np.sum(W,axis = 1)
                                    sensImage[idx1] += GW*AN[j,i]
                                    sensImage[idx2] += GW*AN[j,i+q]
                                else:
                                    sensImage[idx1] += G*AN[j,i]
                                    sensImage[idx2] += G*AN[j,i+q]
                if Flag:
                    if np.any(psf!=0): 
                        sensImageSubs[:,sub] = self.gaussFilter(sensImage,psf)+1e-5
                    else:
                        sensImageSubs[:,sub] = sensImage+1e-5
                if np.any(psf!=0):
                    backProjImage = self.gaussFilter(backProjImage,psf)
                img_sens = sensImageSubs[:,sub]
                betaj = beta * wj /img_sens
                img_em = imgOld*backProjImage/img_sens   
                img = img_em
                img_reg = 1/(2*wj) * prior.imCropUndo((W*prior.Div(imgOld)).sum(axis=1))
            
                #img_reg = imgOld - 1/(2*wj)*prior.GradT(W*prior.Grad(imgOld.reshape(matrixSize[:2],order='F'))).flatten('F')
                img = 2*img_em/(np.sqrt((1-betaj*img_reg)**2+4*betaj*img_em)+(1-betaj*img_reg)+1e-5)
                
                
            if display:
                plt.imshow(img.reshape(matrixSize[:2],order='F'),vmin=0,vmax=3.5),plt.title('img',fontsize=15)
                plt.pause(0.15)
            Flag = False
            sensImage = 0
        img = np.reshape(img,matrixSize[:2],order='F')                
        #img_reg = np.reshape(img_reg,matrixSize[:2],order='F')                
        print('reconstructed in: {} min.'.format((time.time()-tic)/60))                    
        return img
    
        
    def iSensImageBatch2D(self, AN=None, nsubs = 1, psf=0):
        
        if AN is None:
            batch_size = 1
            AN = np.ones([batch_size, self.sinogram.nRadialBins,self.sinogram.nAngularBins],dtype='float')
        else:
             if np.ndim(AN)==2:
                  AN = AN[None,:,:]
             batch_size = AN.shape[0]
             
        sensImageSubBatch = np.zeros([batch_size,nsubs,self.image.matrixSize[0]*self.image.matrixSize[1]],dtype='float')
        sensImage = np.zeros([batch_size,np.prod(self.image.matrixSize[:2])],dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2
        [numAng,subSize] = self.angular_subsets(nsubs)
        
        for sub in range(nsubs):
            sensImage = 0*sensImage
            for ii in range(subSize//2):
                i = numAng[ii,sub]
                for j in range(self.sinogram.nRadialBins):
                    M0 = self.geoMatrix[0][i,j]
                    if not np.isscalar(M0):
                        M = M0[:,0:3].astype('int32')
                        G = M0[:,3]/1e4
                        idx1 = M[:,0] + M[:,1]*matrixSize[0]
                        idx2 = M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])

                        if self.scanner.isTof:
                            W = self.tofMatrix[0][i,j]/1e4
                            GW = G*np.sum(W,axis = 1)
                            for b in range(batch_size):
                                sensImage[b,idx1] += GW*AN[b,j,i]
                                sensImage[b,idx2] += GW*AN[b,j,i+q]
                        else:
                            for b in range(batch_size):
                                sensImage[b,idx1] += G*AN[b,j,i]
                                sensImage[b,idx2] += G*AN[b,j,i+q]
            for b in range(batch_size):
                sensImageSubBatch[b,sub,:] = self.gaussFilter(sensImage[b,:],psf)*self.mask_fov()  
        iSensImageSubBatch = 1/(sensImageSubBatch)
        iSensImageSubBatch[np.isinf(iSensImageSubBatch)]=0
        if batch_size==1:
             iSensImageSubBatch = iSensImageSubBatch[0,:,:]
        return iSensImageSubBatch
    
    def OSEM2D(self, prompts,img=None,RS=None, AN=None, iSensImg = None, niter=100, nsubs=1,  tof=False, psf=0):
    
        # sinodata dimensions: (batch_size, nRadialBins, nAngularBins, totalNumberOfSinogramPlanes, nTofBins)
        # images dimensions: (batch_size,nColums,nRows,nSlices)
        import time
        tic = time.time()
        [numAng,subSize] = self.angular_subsets(nsubs)
        if tof and not self.scanner.isTof:
               raise ValueError("The scanner is not TOF") 
        if not tof and np.ndim(prompts)==2:
             batch_size = 1
             prompts = prompts[None,:,:]                 
        elif tof and np.ndim(prompts)==3:
              batch_size = 1
              prompts = prompts[None,:,:,:]
        else:
             batch_size = prompts.shape[0]

        if img is None:
            img =  np.ones([batch_size,np.prod(self.image.matrixSize[:2])],dtype='float')
        else:
            if batch_size>1 and img.shape[0]!=batch_size:
                raise ValueError("1st img dimension dosn't match batch_size")
        nVoxls = np.prod(self.image.matrixSize[:2])
        img = np.reshape(img,[batch_size,nVoxls],order='F')
        
        if RS is None:
            RS = 0*prompts
        if AN is None:
            AN = np.ones([batch_size,self.sinogram.nRadialBins,self.sinogram.nAngularBins],dtype='float')
        else:
            if np.ndim(AN)==2:
                 AN = AN[None,:,:]
        if iSensImg is None:
            iSensImg = self.iSensImageBatch2D(AN, nsubs, psf)
        if np.ndim(iSensImg)==2:
            iSensImg = iSensImg[None,:,:]
                 

        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2

        
        for n in range(niter):
            for sub in range(nsubs):
                img_ = img.copy() # to make sure "img" is immutable
                if np.any(psf!=0): 
                    for b in range(batch_size):
                        img_[b,:] = self.gaussFilter(img_[b,:],psf)
                backProjImage = 0*img_ 
                for ii in range(subSize//2):
                    i = numAng[ii,sub]
                    for j in range(self.sinogram.nRadialBins):
                        M0 = self.geoMatrix[0][i,j]
                        if not np.isscalar(M0):
                            M = M0[:,0:3].astype('int32')
                            G = M0[:,3]/1e4
                            idx1 = M[:,0] + M[:,1]*matrixSize[0]
                            idx2 = M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])
                            if tof:
                                W = self.tofMatrix[0][i,j]/1e4
                                for b in range(batch_size):
                                    backProjImage[b,idx1] += G*AN[b,j,i]*W.dot(prompts[b,j,i,:]/((AN[b,j,i]*G*img_[b,idx1]).dot(W)+RS[b,j,i,:]+1e-5))
                                    backProjImage[b,idx2] += G*AN[b,j,i+q]*W.dot(prompts[b,j,i+q,:]/((AN[b,j,i+q]*G*img_[b,idx2]).dot(W)+RS[b,j,i+q,:]+1e-5))
                            else:
                                for b in range(batch_size):
                                    backProjImage[b,idx1] += G*AN[b,j,i]*(prompts[b,j,i]/(AN[b,j,i]*(G.dot(img_[b,idx1]))+RS[b,j,i]+1e-5))
                                    backProjImage[b,idx2] += G*AN[b,j,i+q]*(prompts[b,j,i+q]/(AN[b,j,i+q]*(G.dot(img_[b,idx2]))+RS[b,j,i+q]+1e-5))
                if np.any(psf!=0):
                    for b in range(batch_size):
                        backProjImage[b,:] = self.gaussFilter(backProjImage[b,:],psf)
                img = img*backProjImage*iSensImg[:,sub,:]
        
        img = np.reshape(img,[batch_size,matrixSize[0],matrixSize[1]],order='F')
        if batch_size == 1:
             img = img[0,:,:]
        print(f'{batch_size} batches reconstructed in: {(time.time()-tic):.3f} sec.')                      
        return img
   
    def forwardDivideBackwardBatch2D(self, imgb, prompts, RS=None, AN=None, nsubs=1, subset_i=0, tof=False, psf=0):
    
        # 3D sinodata dimensions: (batch_size, nRadialBins, nAngularBins, totalNumberOfSinogramPlanes, nTofBins)
        # 3D images dimnesions: (batch_size,nColums,nRows,nSlices)
        #import time
        [numAng,subSize] = self.angular_subsets(nsubs)
        if nsubs>1 and subset_i>(nsubs-1):
            raise ValueError(f"subset_i must be in [0, {nsubs}]")
        if tof and not self.scanner.isTof:
               raise ValueError("The scanner is not TOF") 
        batch_size = prompts.shape[0]
        img = imgb.copy()
        if img.shape[0]!=batch_size:
            raise ValueError("1st img dimension dosn't match batch_size")
        nVoxls = np.prod(self.image.matrixSize[:2])
        img = np.reshape(img,[batch_size,nVoxls],order='F') 
        if np.ndim(prompts)!=4:
            tof = False
        if RS is None:
            RS = 0*prompts
        if AN is None:
            AN = np.ones([batch_size,self.sinogram.nRadialBins,self.sinogram.nAngularBins],dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2
        #nTrimmedRadialBins = np.arange(self.sinogram.nRadialBins//6,self.sinogram.nRadialBins-self.sinogram.nRadialBins//6)
        if np.any(psf!=0):
            for b in range(batch_size):
                img[b,:] = self.gaussFilter(img[b,:],psf)
        backProjImage = 0*img 
        for ii in range(subSize//2):
            i = numAng[ii,subset_i]
            for j in range(self.sinogram.nRadialBins):#nTrimmedRadialBins:
                M0 = self.geoMatrix[0][i,j]
                if not np.isscalar(M0):
                    M = M0[:,0:3].astype('int32')
                    G = M0[:,3]/1e4
                    idx1 = M[:,0] + M[:,1]*matrixSize[0]
                    idx2 = M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])
                    if tof:
                        W = self.tofMatrix[0][i,j]/1e4
                        for b in range(batch_size):
                            backProjImage[b,idx1] += G*AN[b,j,i]*W.dot(prompts[b,j,i,:]/((AN[b,j,i]*G*img[b,idx1]).dot(W)+RS[b,j,i,:]+1e-5))
                            backProjImage[b,idx2] += G*AN[b,j,i+q]*W.dot(prompts[b,j,i+q,:]/((AN[b,j,i+q]*G*img[b,idx2]).dot(W)+RS[b,j,i+q,:]+1e-5))
                    else:
                        for b in range(batch_size):
                            backProjImage[b,idx1] += G*AN[b,j,i]*(prompts[b,j,i]/(AN[b,j,i]*G.dot(img[b,idx1])+RS[b,j,i]+1e-5))
                            backProjImage[b,idx2] += G*AN[b,j,i+q]*(prompts[b,j,i+q]/(AN[b,j,i+q]*G.dot(img[b,idx2])+RS[b,j,i+q]+1e-5))
        if np.any(psf!=0):
            for b in range(batch_size):
                backProjImage[b,:] = self.gaussFilter(backProjImage[b,:],psf)*self.mask_fov()
                 
        #print(f'{batch_size} batches forwad-backprojected in: {(time.time()-tic)/60:.3f} min.')                      
        return backProjImage

    def forwardBackwardBatch2D(self, imgb, RS=None, AN=None, nsubs=1, subset_i=0, tof=False, psf=0):
    
        # 3D sinodata dimensions: (batch_size, nRadialBins, nAngularBins, totalNumberOfSinogramPlanes, nTofBins)
        # 3D images dimnesions: (batch_size,nColums,nRows,nSlices)
        #import time
        [numAng,subSize] = self.angular_subsets(nsubs)
        if nsubs>1 and subset_i>(nsubs-1):
            raise ValueError(f"subset_i must be in [0, {nsubs}]")
        if tof and not self.scanner.isTof:
               raise ValueError("The scanner is not TOF") 

        img = imgb.copy()
        if np.ndim(img)==2:
            batch_size =1
        else:
            batch_size = img.shape[0]

        nVoxls = np.prod(self.image.matrixSize[:2])
        img = np.reshape(img,[batch_size,nVoxls],order='F') 
        if RS is None:
            dims = (batch_size,self.sinogram.nRadialBins,self.sinogram.nAngularBins)
            if tof and self.scanner.isTof: dims+=(self.sinogram.nTofBins,)
            RS = np.zeros(dims,dtype='float')
        if AN is None:
            AN = np.ones([batch_size,self.sinogram.nRadialBins,self.sinogram.nAngularBins],dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2
        #nTrimmedRadialBins = np.arange(self.sinogram.nRadialBins//6,self.sinogram.nRadialBins-self.sinogram.nRadialBins//6)
        if np.any(psf!=0):
            for b in range(batch_size):
                img[b,:] = self.gaussFilter(img[b,:],psf)
        backProjImage = 0*img 
        for ii in range(subSize//2):
            i = numAng[ii,subset_i]
            for j in range(self.sinogram.nRadialBins):#nTrimmedRadialBins:
                M0 = self.geoMatrix[0][i,j]
                if not np.isscalar(M0):
                    M = M0[:,0:3].astype('int32')
                    G = M0[:,3]/1e4
                    idx1 = M[:,0] + M[:,1]*matrixSize[0]
                    idx2 = M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])
                    if tof:
                        W = self.tofMatrix[0][i,j]/1e4
                        for b in range(batch_size):
                            backProjImage[b,idx1] += G*AN[b,j,i]*W.dot((AN[b,j,i]*G*img[b,idx1]).dot(W)+RS[b,j,i,:])
                            backProjImage[b,idx2] += G*AN[b,j,i+q]*W.dot((AN[b,j,i+q]*G*img[b,idx2]).dot(W)+RS[b,j,i+q,:])
                    else:
                        for b in range(batch_size):
                            backProjImage[b,idx1] += G*AN[b,j,i]*(AN[b,j,i]*G.dot(img[b,idx1])+RS[b,j,i])
                            backProjImage[b,idx2] += G*AN[b,j,i+q]*(AN[b,j,i+q]*G.dot(img[b,idx2])+RS[b,j,i+q])
        if np.any(psf!=0):
            for b in range(batch_size):
                backProjImage[b,:] = self.gaussFilter(backProjImage[b,:],psf)*self.mask_fov()
        backProjImage = np.reshape(backProjImage,[batch_size,matrixSize[0],matrixSize[1]],order='F')
        if batch_size == 1:
             backProjImage = backProjImage[0]         
        #print(f'{batch_size} batches forwad-backprojected in: {(time.time()-tic)/60:.3f} min.')                      
        return backProjImage

    def backwardBatch2D_i(self, prompts, AN=None, nsubs=1, tof=False, psf=0):
    
        # 3D sinodata dimensions: (batch_size, nRadialBins, nAngularBins, totalNumberOfSinogramPlanes, nTofBins)
        # 3D images dimnesions: (batch_size,nColums,nRows,nSlices)
        #import time
        [numAng,subSize] = self.angular_subsets(nsubs)

        if tof and not self.scanner.isTof:
               raise ValueError("The scanner is not TOF") 
        if np.ndim(prompts)==4 or (np.ndim(prompts)==3 and not self.scanner.isTof):
            batch_size = prompts.shape[0]
        else:
            batch_size = 1

        if batch_size == 1:
            prompts = prompts[None]
               

        nVoxls = np.prod(self.image.matrixSize[:2])

        if AN is None:
            AN = np.ones([batch_size,self.sinogram.nRadialBins,self.sinogram.nAngularBins],dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2
        #nTrimmedRadialBins = np.arange(self.sinogram.nRadialBins//6,self.sinogram.nRadialBins-self.sinogram.nRadialBins//6)

        backProjImage = np.zeros([batch_size,nVoxls,nsubs],dtype='float')
        for s in range(nsubs):
            for ii in range(subSize//2):
                i = numAng[ii,s]
                for j in range(self.sinogram.nRadialBins):#nTrimmedRadialBins:
                    M0 = self.geoMatrix[0][i,j]
                    if not np.isscalar(M0):
                        M = M0[:,0:3].astype('int32')
                        G = M0[:,3]/1e4
                        idx1 = M[:,0] + M[:,1]*matrixSize[0]
                        idx2 = M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])
                        if tof:
                            W = self.tofMatrix[0][i,j]/1e4
                            for b in range(batch_size):
                                backProjImage[b,idx1,s] += G*AN[b,j,i]*W.dot(prompts[b,j,i,:])
                                backProjImage[b,idx2,s] += G*AN[b,j,i+q]*W.dot(prompts[b,j,i+q,:])
                        else:
                            for b in range(batch_size):
                                backProjImage[b,idx1,s] += G*AN[b,j,i]*(prompts[b,j,i])
                                backProjImage[b,idx2,s] += G*AN[b,j,i+q]*(prompts[b,j,i+q])
            if np.any(psf!=0):
                for b in range(batch_size):
                    backProjImage[b,:,s] = self.gaussFilter(backProjImage[b,:,s],psf)*self.mask_fov()
                 
        backProjImage = np.reshape(backProjImage,[batch_size,matrixSize[0],matrixSize[1],nsubs],order='F')
        # if nsubs==1:
        #     backProjImage = backProjImage[:,:,:,0]
        if batch_size == 1:
             backProjImage = backProjImage[0]    
                  
        return backProjImage


    def mask_fov(self,reconFovRadious=None):
        if self.fov_mask is None:
            if reconFovRadious is not None:
                 reconRadious = reconFovRadious*0.96
            elif hasattr(self.image,'reconFovRadious'):
                 reconRadious = self.image.reconFovRadious*0.96
            else:
                 reconRadious = self.sinogram.nRadialBins*self.scanner.xCrystalDimCm/4*0.96
            x = self.image.voxelSizeCm[0]* np.arange(-self.image.matrixSize[0]//2, self.image.matrixSize[0]//2, 1)
            y = self.image.voxelSizeCm[1]*np.arange(-self.image.matrixSize[1]//2, self.image.matrixSize[1]//2, 1)
            xx, yy = np.meshgrid(x, y)     
            
            if self.is3d:
                 mask = np.zeros(self.image.matrixSize)
                 for i in range(mask.shape[2]):
                      mask[:,:,i] = (xx**2 + yy**2)<reconRadious**2
            else:
                mask = ((xx**2 + yy**2)<reconRadious**2).flatten('F')
            self.fov_mask = mask
        else:
            mask = self.fov_mask
        return mask
    
    def OSEM3D_python(self, prompts,img=None,RS=None,niter=100, nsubs=1, AN=None, psf=0):
        import time
        [numAng,subSize] = self.angular_subsets(nsubs)
        if img is None:
            img = np.ones(self.image.matrixSize,dtype='float')
        img = img.flatten('F')  
        sensImage = np.zeros_like(img)
        sensImageSubs = np.zeros((np.prod(self.image.matrixSize),nsubs),dtype='float')
        if RS is None:
            RS = 0*prompts
        if AN is None:
            AN = np.ones([self.sinogram.nRadialBins,self.sinogram.nAngularBins,self.sinogram.totalNumberOfSinogramPlanes],dtype='float')
         
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins//2
        Flag = True    
        
        nUniqueAxialPlanes = len(self.sinogram.uniqueAxialPlanes)
        allPlanes = []
        for i in range(len(self.sinogram.uniqueAxialPlanes)):
            allPlanes.append(np.nonzero(self.sinogram.planeMirrorTranslation[:,0] == i+1)[0])
        planeMirrorTranslation = self.sinogram.planeMirrorTranslation
        prod = lambda x, y: x.reshape(-1,1).dot(y.reshape(1,-1)).T.astype('int32')
        
        tic = time.time()
        for n in range(niter):
             for sub in range(nsubs):
                 imgOld = self.gaussFilter(img,psf,True)
                 backProjImage = 0*img
                 for ii in range(subSize//2):
                     i = numAng[ii,sub]
                     for j in range(self.sinogram.nRadialBins):
                         for p in range(nUniqueAxialPlanes):#
                             M0 = self.geoMatrix[p][i,j]
                             if not np.isscalar(M0):
                                 M = M0[:,0:3].astype('int32')
                                 G = M0[:,3]/1e4
                                 H = planeMirrorTranslation[allPlanes[p],:]
                                 idxAxial = matrixSize[0]*matrixSize[1]*(prod(H[:,1],M[:,2]) + H[:,2])
                                 idx1 = (M[:,0] + M[:,1]*matrixSize[0]).reshape(-1,1) + idxAxial
                                 idx2 = (M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])).reshape(-1,1) + idxAxial
                                 
                                 an = AN[j,i,allPlanes[p]]
                                 f = an*(prompts[j,i,allPlanes[p]]/(an*G.dot(imgOld[idx1])+RS[j,i,allPlanes[p]]+1e-5))                                 
                                 backProjImage[idx1] += G.reshape(-1,1).dot(f.reshape(1,-1))
                                 an = AN[j,i+q,allPlanes[p]]
                                 f = an*(prompts[j,i+q,allPlanes[p]]/(an*G.dot(imgOld[idx2])+RS[j,i+q,allPlanes[p]]+1e-5))
                                 backProjImage[idx2] += G.reshape(-1,1).dot(f.reshape(1,-1)) 
                                 if Flag:
                                      sensImage[idx1] += G.reshape(-1,1).dot(AN[j,i,allPlanes[p]].reshape(1,-1)) 
                                      sensImage[idx2] += G.reshape(-1,1).dot(AN[j,i+q,allPlanes[p]].reshape(1,-1))
                 if Flag:
                      sensImageSubs[:,sub] = sensImage+1e-5
                 backProjImage = self.gaussFilter(backProjImage,psf)
                 img = imgOld*backProjImage/(sensImageSubs[:,sub])
             Flag = False
             sensImage = 0                     
        print('forward-projected in: {} min.'.format((time.time()-tic)/60))                    
        return img.reshape(matrixSize,order='F')



    """
    ******************************************************************************************************************
    
                                                 Functions for 3D reconstruction using April
    
    ******************************************************************************************************************
     
    """   
    def createConfigFile(self,flname,input_file, output_data_flname, output_filename, gpu = True, project_mode = True, nsubs = 1, subsetIndex = 0):
          
          f = open(flname,"w+")
          if project_mode:
               f.write("Projection Parameters :=\n" \
                       "output type := Sinogram3DSiemensMmr\n")
               if gpu:
                    f.write("projector := CuSiddonProjector\n" \
                            "projector block size := {256,1,1}\n" \
                            "gpu id := 0\n")
               else:
                    f.write("projector := Siddon\n")
               f.write(f"output projection := {output_data_flname}\n")
          else:
               f.write("Backproject Parameters :=\n" \
                       "input type := Sinogram3DSiemensMmr\n")
               if gpu:
                    f.write("backprojector := CuSiddonProjector\n" \
                            "backprojector block size := {576,1,1}\n" \
                            "gpu id := 0\n")
               else:
                    f.write("backprojector := Siddon\n")
               f.write(f"output image := {output_data_flname}\n")
     
          f.write("siddon number of samples on the detector := 1\n" \
                  "siddon number of axial samples on the detector := 1\n")
          if nsubs>1:
               f.write(f"number of subsets := {nsubs}\n")
               f.write(f"subset index := {subsetIndex}\n")
          
          f.write(f"input file := {input_file}\n")
          f.write(f"output filename := {output_filename}\n")
          f.close()     
          return 

    def write_to_apirl(self,flname, data=None, sino_mode=False):
          if data is None:
               dataType = 'short float'
               itemsize = 4
          else:
               if data.dtype.name == 'int32':
                   dataType = 'signed integer' 
               elif data.dtype.name == 'int16':
                    dataType = 'signed integer'              
               elif data.dtype.name == 'float64':
                    dataType = 'long float'
               elif data.dtype.name == 'float32':
                    dataType = 'short float'               
               itemsize = data.itemsize
   
          f = open(flname+".h33","w+") 
          f.write("!INTERFILE :=\n")
          f.write(f"!name of data file := {flname}.i33\n")
          if sino_mode:
               f.write(f"!number format := {dataType}\n")
               f.write(f"!number of bytes per pixel := {itemsize}\n")
               f.write("imagedata byte order := LITTLEENDIAN\n" \
               "number of dimensions := 4\n" \
               "matrix axis label [4] := segment\n" \
               "!matrix size [4] := 11\n" \
               "matrix axis label [2] := view\n" \
               "!matrix size [2] := 252\n" \
               "matrix axis label [3] := axial coordinate\n" \
               "!matrix size [3] := { 127, 115, 115, 93, 93, 71, 71, 49, 49, 27, 27 }\n" \
               "matrix axis label [1] := tangential coordinate\n")
               f.write(f"!matrix size [1] := {self.sinogram.nRadialBins}\n")
               f.write("minimum ring difference per segment := {  -5, 6, -16, 17, -27, 28, -38, 39, -49, 50, -60 }\n" \
               "maximum ring difference per segment := {  5, 16, -6, 27, -17, 38, -28, 49, -39, 60, -50 }\n" \
               "number of rings := 64\n" \
               "!END OF INTERFILE :=\n")
          else:
               matrix_size = data.shape
               f.write(f"!total number of images := {matrix_size[2]}\n")
               f.write("!imagedata byte order := LITTLEENDIAN\n")
               f.write(f"!matrix size [1] := {matrix_size[0]}\n")
               f.write(f"!matrix size [2] := {matrix_size[1]}\n")
               f.write(f"!matrix size [3] := {matrix_size[2]}\n")
               f.write("!number format := float\n"
               "!number of bytes per pixel := 4\n" \
               "scaling factor (mm/pixel) [1] := 2.086260\n" \
               "scaling factor (mm/pixel) [2] := 2.086260\n" \
               "scaling factor (mm/pixel) [3] := 2.031250\n" \
               "!END OF INTERFILE :=\n")
          f.close()   
          f = open(flname+".i33","wb")       
          if data is not None:
                f.write(data.tobytes(order='F'))
          f.close()
     

    def gaussFilterBatch(self,img,fwhm):
         voxelSizeCm = self.image.voxelSizeCm
         is3d = self.is3d
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
    
    def simulateSinogramData(self,img, mumap = None, AF = None, NF = None, counts= 1e7, psf = 0, tof = False, randomsFraction = 0):
        #2D/3D images dimensions: (batch_size,nColums,nRows,[nSlices])
        if self.is3d:
             if np.ndim(img)==3:
                  batch_size = 1
             else:
                  batch_size=img.shape[0]
             projector = lambda x: self.forwardProjectBatch3D(x, psf=psf)
        else:
             if np.ndim(img)==2:
                  batch_size = 1
             else:
                  batch_size = img.shape[0]
             projector = lambda x: self.forwardProjectBatch2D(x, psf=psf)
     
        if AF is None:
             if mumap is None:
                  AF = 1
             else:
                  if np.ndim(mumap)!=np.ndim(img):
                       raise ValueError('mumap must have the same size as input img')
                  AF = np.exp(-projector(mumap*self.image.voxelSizeCm[0]))
                  AF[np.isinf(AF)] = 0  
             
        if NF is None:
             _,_,gaps = self.LorsTransaxialCoor()
             gaps = ~gaps.astype('bool')
             NF = np.zeros_like(AF)
             if batch_size == 1:
                  NF = NF[None]             
             if self.is3d:
                  for b in range(batch_size):
                       for i in range(self.sinogram.totalNumberOfSinogramPlanes):
                            NF[b,:,:,i] = gaps*np.exp(-1*np.random.rand(self.sinogram.nRadialBins,self.sinogram.nAngularBins))
             else:
                  for b in range(batch_size):
                       NF[b,:,:] = gaps*np.exp(-1*np.random.rand(self.sinogram.nRadialBins,self.sinogram.nAngularBins))
             if batch_size == 1:
                  NF = NF[0]

        if np.isscalar(counts):
             counts = counts*np.ones(batch_size,)                   
        truesFraction = 1 - randomsFraction
        
        y = projector(img)
        y_att = y*AF
        y_poisson = np.zeros_like(y)
        
        if batch_size>1:
            for b in range(batch_size):
                if self.is3d:
                     scale_factor = counts[b]*truesFraction/y_att[b,:,:,:].sum()
                     y_poisson[b,:,:,:] = np.random.poisson(y_att[b,:,:,:]*scale_factor)/scale_factor                
                else:
                     scale_factor = counts[b]*truesFraction/y_att[b,:,:].sum()
                     y_poisson[b,:,:] = np.random.poisson(y_att[b,:,:]*scale_factor)/scale_factor
        else:
                scale_factor = counts*truesFraction/y_att.sum()
                y_poisson = np.random.poisson(y_att*scale_factor)/scale_factor
        if randomsFraction!=0:
            Randoms = np.zeros_like(y)
            r_poisson = np.ones_like(y)
            if batch_size>1:
                for b in range(batch_size):
                    if self.is3d:
                         scale_factor_randoms = counts[b]*randomsFraction/r_poisson[b,:,:,:].sum()
                         r_poisson[b,:,:,:] = np.random.poisson(r_poisson[b,:,:,:]*scale_factor_randoms)/scale_factor_randoms                    
                    else:
                         scale_factor_randoms = counts[b]*randomsFraction/r_poisson[b,:,:].sum()
                         r_poisson[b,:,:] = np.random.poisson(r_poisson[b,:,:]*scale_factor_randoms)/scale_factor_randoms
            else:
                scale_factor_randoms = counts*randomsFraction/r_poisson.sum()
                r_poisson = np.random.poisson(r_poisson*scale_factor_randoms)/scale_factor_randoms
        else:
            Randoms = 0   
        prompts = y_poisson*NF + Randoms
        return prompts, AF,NF, Randoms     

    def reserve_temPath(self,batch_size):
     
          tmpath = self.engine.temPath
          if type(tmpath)!=list:
               temPaths = [tmpath]
          else:
               if len(tmpath)>=batch_size:
                    self.engine.temPath = tmpath[0:batch_size]
                    return 
               else:
                    tmpath = tmpath[0]
                    temPaths = [tmpath]
     
          for f in range(batch_size-1):
                temPath = tmpath+str(f)
                if not os.path.exists(temPath):
                     os.makedirs(temPath)
                temPaths.append(temPath)
          self.engine.temPath = temPaths
          return            
   
    def fwd_subprocess(self, img_b, input_img_flname, out_sino_flname, config_flname, nsubs, subsetIndex, psf):
          sino_shape = self.sinogram.shape
          if np.any(psf!=0):
               img_b = self.gaussFilterBatch(img_b,psf)
          self.write_to_apirl(input_img_flname,img_b.transpose(1,0,2))
          subprocess.run([self.engine.binPath +self.engine.bar+'project', config_flname], stdout=subprocess.PIPE);            
          tmp = np.fromfile(out_sino_flname+'.i33',dtype='float32')
          if nsubs ==1:
               sino = tmp.reshape(sino_shape, order='F')
          else:
               subshape = [sino_shape[0],sino_shape[1]//nsubs,sino_shape[2]]
               tmp = tmp.reshape(subshape, order='F')
               sino = np.zeros(sino_shape,dtype='float32')
               sino[:,subsetIndex::nsubs,:] = tmp
          sino[np.isnan(sino)]=0
          os.remove(out_sino_flname+'.h33')
          os.remove(out_sino_flname+'.i33')
          os.remove(input_img_flname+'.h33')
          os.remove(input_img_flname+'.i33')     
          return sino

    def forwardProjectBatch3D(self,img, nsubs = 1, subsetIndex=0, psf = 0):
          if nsubs>1:
               self.check_nsubs(nsubs)
          
          img = img.astype('float32')      
          if np.ndim(img)==3:
               batch_size = 1
               img = img[None,:,:,:]
          else:
               batch_size = img.shape[0]
               
          sino_shape = self.sinogram.shape         
          sinoOut = np.zeros((batch_size, sino_shape[0], sino_shape[1], sino_shape[2]),dtype = 'float32')
          out_sino_flname = []
          sample_sino_flname =[]
          input_img_flname = []
          config_flname = []
          
          num_temp = batch_size if (self.engine.multiprocess and self.engine.gpu) else 1
          self.reserve_temPath(num_temp)
          for b in range(num_temp):
               out_sino_flname.append(self.engine.temPath[b] + self.engine.bar + 'out_sino')
               sample_sino_flname.append(self.engine.temPath[b] + self.engine.bar + 'sample_sino')
               input_img_flname.append(self.engine.temPath[b] + self.engine.bar + 'input_img')
               config_flname.append(self.engine.temPath[b] + self.engine.bar + 'fwdproj.par')             
               if not os.path.exists(sample_sino_flname[b]+'.h33'):
                    self.write_to_apirl(sample_sino_flname[b], sino_mode=True)
               self.createConfigFile(config_flname[b],input_img_flname[b]+'.h33', sample_sino_flname[b]+'.h33', out_sino_flname[b], self.engine.gpu, True, nsubs, subsetIndex)

          if num_temp>1: # multi-processing is only usefull for GPU version
               pool = mp.Pool(processes=batch_size)
               sino = pool.starmap(self.fwd_subprocess, [(img[b,:,:,:], input_img_flname[b], out_sino_flname[b], config_flname[b], nsubs, subsetIndex, psf) for b in range(batch_size)])
               pool.close()
               for b in range(batch_size):
                    sinoOut[b,:,:,:] = sino[b]
          else:
               for b in range(batch_size):
                    sinoOut[b,:,:,:] = self.fwd_subprocess(img[b,:,:,:], input_img_flname[0], out_sino_flname[0], config_flname[0], nsubs, subsetIndex, psf)
          if batch_size==1:
               sinoOut = sinoOut[0,:,:,:]         
          return sinoOut


    def bkd_subprocess(self,sino_b, input_sino_flname, out_img_flname, config_flname, nsubs, subsetIndex, psf):
          matrix_size = self.image.matrixSize
          self.write_to_apirl(input_sino_flname, sino_b, True) 
          subprocess.run([self.engine.binPath +self.engine.bar+'backproject', config_flname],stdout=subprocess.PIPE);
          img = np.fromfile(out_img_flname+'.i33',dtype='float32').reshape(matrix_size, order='F').transpose(1,0,2)
          img[np.isnan(img)]=0
          if np.any(psf!=0):
               img = self.gaussFilterBatch(img,psf)
          os.remove(out_img_flname+'.h33')
          os.remove(out_img_flname+'.i33')
          os.remove(input_sino_flname+'.h33')
          os.remove(input_sino_flname+'.i33')
     
          return img*self.mask_fov()

    def backProjectBatch3D(self,sino, nsubs = 1, subsetIndex=0, psf = 0):
     
          sino = sino.astype('float32')      
          if np.ndim(sino)==3:
               batch_size = 1
               sino = sino[None,:,:,:]
          else:
               batch_size = sino.shape[0]
  
          matrix_size = self.image.matrixSize
          imgOut = np.zeros((batch_size, matrix_size[0], matrix_size[1], matrix_size[2]),dtype = 'float32')
          out_img_flname =[]
          sample_img_flname =[]
          input_sino_flname =[]
          config_flname =[]
          
          num_temp = batch_size if (self.engine.multiprocess and self.engine.gpu) else 1
          self.reserve_temPath(num_temp)
          for b in range(num_temp):
               out_img_flname.append(self.engine.temPath[b] + self.engine.bar + 'out_img')
               sample_img_flname.append(self.engine.temPath[b] + self.engine.bar + 'sample_img')
               input_sino_flname.append(self.engine.temPath[b] + self.engine.bar + 'input_sino')
               config_flname.append(self.engine.temPath[b] + self.engine.bar + 'backproj.par')
               self.createConfigFile(config_flname[b],input_sino_flname[b]+'.h33', sample_img_flname[b]+'.h33', out_img_flname[b],
                                self.engine.gpu, False, nsubs, subsetIndex)
               if not os.path.exists(sample_img_flname[b]+'.h33'):
                    self.write_to_apirl(sample_img_flname[b],np.zeros(matrix_size,dtype='float32')) 
          
          if num_temp>1: # multi-processing is only usefull for GPU version
               pool = mp.Pool(processes=batch_size)
               img = pool.starmap(self.bkd_subprocess, [(sino[b,:,:,:], input_sino_flname[b], out_img_flname[b], config_flname[b], nsubs, subsetIndex, psf) for b in range(batch_size)])
               pool.close()
               for b in range(batch_size):
                    imgOut[b,:,:,:] = img[b]
          else:
               for b in range(batch_size):
                    imgOut[b,:,:,:] = self.bkd_subprocess(sino[b,:,:,:], input_sino_flname[0], out_img_flname[0], config_flname[0], nsubs, subsetIndex, psf)      
          if batch_size==1:
               imgOut = imgOut[0,:,:,:]
          return imgOut       

    def iSensImageBatch3D(self, AN = None, nsubs = 1, psf = 0):
          if AN is None:
               AN = np.ones(self.sinogram.shape,dtype='float32')
          if np.ndim(AN)==3:
               batch_size = 1
               AN = AN[None,:,:,:]
          else:
               batch_size = AN.shape[0]
          sensImgOut = np.zeros((batch_size,nsubs, self.image.matrixSize[0], self.image.matrixSize[1], self.image.matrixSize[2]),dtype='float32')
     
          for i in range(nsubs):
               sensImgOut[:,i,:,:,:] = self.backProjectBatch3D(AN,nsubs = nsubs, subsetIndex=i, psf=psf)
     
          if batch_size==1:
               sensImgOut = sensImgOut[0,:,:,:,:]
          sensImgOut = 1/sensImgOut
          sensImgOut[np.isinf(sensImgOut)]=0
          return sensImgOut

    def forwardDivideBackwardBatch3D(self,img, prompts, RS, AN, nsubs, subsetIndex, psf):
         if RS is None: RS = 0
         y = self.forwardProjectBatch3D(img,nsubs = nsubs, subsetIndex=subsetIndex, psf=psf) + RS + 1e-4
         out = self.backProjectBatch3D(prompts/y,nsubs = nsubs, subsetIndex=subsetIndex, psf=psf)
         out[np.isinf(out)]=0
         return out

    def OSEM3D(self,prompts, AN = None, RS = None, iSensImg = None, img = None, niter = 1, nsubs = 1, psf=0, display = False):
          #import time
          if np.ndim(prompts)==3:
               batch_size = 1
          else:
               batch_size = prompts.shape[0]
          if AN is None:
               if batch_size>1:
                    sino_shape = [batch_size,self.sinogram.shape[0],self.sinogram.shape[1],self.sinogram.shape[2]]
               else:
                    sino_shape = self.sinogram.shape
               AN = np.ones(sino_shape,dtype='float32') 
          if iSensImg is None:
               #print(f'Calculate sensitivity image for {batch_size} sinograms and {nsubs} subsets\n')
               iSensImg = self.iSensImageBatch3D(AN,nsubs,psf) 
          if img is None:
               if batch_size>1:
                    matrix_size = [batch_size, self.image.matrixSize[0],self.image.matrixSize[1],self.image.matrixSize[2]]
               else:
                    matrix_size = self.image.matrixSize
               img = np.ones(matrix_size,dtype='float32')
          #print(f'OSEM recon of {batch_size} sinograms. {niter} iters, {nsubs} nsubs \n')
          #tic = time.time()
          for n in range(niter):
               if display: print(f"iter: {n}")
               for m in range(nsubs):
                   if batch_size ==1:
                        iSenImg_m = iSensImg[m,:,:,:]
                   else:
                        iSenImg_m = iSensImg[:,m,:,:,:]
                   img = img  * self.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, m, psf)*iSenImg_m
          #print(f'Done in {(time.time()-tic)/60:.3f} min.')
          return img

    def forwardDivideSubtractBackwardBatch3D(self,img, prompts, RS, AN, nsubs, subsetIndex, psf):
         y = self.forwardProjectBatch3D(img,nsubs = nsubs, subsetIndex=subsetIndex, psf=psf) + RS + 1e-4
         out = self.backProjectBatch3D(((prompts/y)-1.0),nsubs = nsubs, subsetIndex=subsetIndex, psf=psf)
         out[np.isinf(out)]=0
         return out
    
    def MAPEM3D(self,prompts, AN = None, RS = 0, iSensImg = None, img = None, niter = 1, nsubs = 1, psf=0, \
                 display = True, beta=1, prior_object = None, neighborhood_size = 3, weight_type ='bowsher', weights = None, prior_img = None, \
                   bowsher_b = 20, gaussian_sigma = 0.2):
          
          # 3D maximum-a-posteriori expectation maximisation using a weighted quadratic prior and De Prior's convexity lemma
          if display: import matplotlib.pyplot as plt
          import time
          if prior_object is None:
               from geometry.Prior import Prior
               prior = Prior(self.image.matrixSize, neighborhood_size)
          else:
               prior = prior_object
          if weights is None:
               if prior_img is None:
                    weights = 1
               else:
                    prior_img_s = self.gaussFilterBatch(prior_img/prior_img.max(),0.25)
                    if weight_type.lower() =='bowsher':
                         weights = prior.BowshserWeights(prior_img_s, bowsher_b)
                    else: # gaussian
                         weights = prior.gaussianWeights(prior_img_s,gaussian_sigma)
          W = (prior.Wd*weights).astype('float32')   
          wj = prior.imCropUndo((W.sum(axis=1)).reshape(prior.imageSizeCrop,order='F'))
          wj_ = 1/wj
          wj_[np.isinf(wj_)]=0
          if np.ndim(prompts)==3:
               batch_size = 1
          else:
               batch_size = prompts.shape[0]
          if AN is None:
               if batch_size>1:
                    sino_shape = [batch_size,self.sinogram.shape[0],self.sinogram.shape[1],self.sinogram.shape[2]]
               else:
                    sino_shape = self.sinogram.shape
               AN = np.ones(sino_shape,dtype='float32') 
     
          if iSensImg is None:
               iSensImg = self.iSensImageBatch3D(AN,nsubs,psf) 

          if img is None:
               if batch_size>1:
                    matrix_size = [batch_size, self.image.matrixSize[0],self.image.matrixSize[1],self.image.matrixSize[2]]
               else:
                    matrix_size = self.image.matrixSize
               img = np.ones(matrix_size,dtype='float32')
          if np.isscalar(beta):
               beta = np.array([beta],dtype='float32')
               if batch_size>1:
                    beta = beta[0]*np.ones(batch_size,dtype='float32')

          tic = time.time()
          for n in range(niter):
               for m in range(nsubs):
                   if batch_size ==1:
                        gamma = beta * wj
                        iS = iSensImg[m,:,:,:]
                        img_em = img  * self.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, m, psf)*iS                   
                        img_reg = img - 1/2*wj_*prior.GradT(W*prior.Grad(img))     
                        img = 2*img_em/((1 - gamma*iS*img_reg) + np.sqrt((1 - gamma*iS*img_reg)**2 + 4*gamma*iS*img_em))
                   else:
                        iS = iSensImg[:,m,:,:,:]
                        img_em = img  * self.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, m, psf)*iS
                        for b in range(batch_size):
                             gamma = beta[b] * wj
                             img_reg = img[b,:,:,:] - 1/2*wj_*prior.GradT(W*prior.Grad(img[b,:,:,:]))
                             img[b,:,:,:] = 2*img_em[b,:,:,:]/((1 - gamma*iS[b,:,:,:]*img_reg) + \
                                np.sqrt((1 - gamma*iS[b,:,:,:]*img_reg)**2 + 4*gamma*iS[b,:,:,:]*img_em[b,:,:,:]))
                   if display: plt.imshow((self.crop_img(img,0.4))[:,:,50],cmap='gist_heat'),plt.pause(0.1),print(f"iter: {n},sub: {m}")
          print(f'Done in {(time.time()-tic)/60:.3f} min.')
          return img
    def removeSampleFiles(self):
          # remove sample files, in case a different projector configuration is used
          temPath = self.engine.temPath if type(self.engine.temPath)==str else self.engine.temPath[0]
          dirname = os.path.dirname(temPath)
          folders = os.listdir(dirname)
          for f in folders:
               fn = dirname + self.engine.bar + f + self.engine.bar
               if os.path.exists(fn+'sample_img.h33'): os.remove(fn+'sample_img.h33')
               if os.path.exists(fn+'sample_img.i33'): os.remove(fn+'sample_img.i33')
               if os.path.exists(fn+'sample_sino.h33'): os.remove(fn+'sample_sino.h33')
               if os.path.exists(fn+'sample_sino.i33'): os.remove(fn+'sample_sino.i33')
          return
    def zeroNanInfs(self,x):
          x[np.isnan(x)]=0
          x[np.isinf(x)]=0
          return x

    """
    ******************************************************************************************************************
    
                                                 Functions for E7 tools
    
    ******************************************************************************************************************
    """
    def get_gaps(self):
          if self.gaps is None:
               _,_,gaps = self.LorsTransaxialCoor()
               gaps2d = ~gaps.astype('bool')
               gaps = np.zeros(self.sinogram.shape,dtype='bool')
               for i in range(gaps.shape[2]):
                    gaps[:,:,i] = gaps2d
               self.gaps = gaps
          return self.gaps     
    def segment_reorder(self):
          centralSegment = (self.sinogram.nSegments)//2
          o = np.zeros([self.sinogram.nSegments,],dtype='int16')
          o[0::2] = np.arange(centralSegment,self.sinogram.nSegments)
          o[1::2] = np.arange(centralSegment-1,-1,-1)
          return o
    def iSSRB(self,sino2d):
          nPlanePerSeg = self.sinogram.numberOfPlanesPerSeg[self.segment_reorder()]
          mo = np.cumsum(nPlanePerSeg)
          no = np.zeros((self.sinogram.nSegments,2),dtype='int16')
          no[:,1] = mo
          no[1:,0] = mo[:-1]
          
          sino3d = np.zeros(self.sinogram.shape,dtype='float32')
          sino3d[:,:,no[0,0]:no[0,1]] = sino2d
          
          for i in range(1,self.sinogram.nSegments,2):
              
              delta = (nPlanePerSeg[0]- nPlanePerSeg[i])//2
              indx = nPlanePerSeg[0] - delta
          
              sino3d[:,:,no[i,0]:no[i,1]] = sino2d[:,:,delta:indx]
              sino3d[:,:,no[i+1,0]:no[i+1,1]] = sino2d[:,:,delta:indx]
          return sino3d
     
    def crop_sino(self,sino):
         if self.sinogram.radialBinCropfactor!=0:
              i =  int(np.ceil(sino.shape[0]*self.sinogram.radialBinCropfactor/2.0)*2)//2
              sinOut = sino[i:sino.shape[0]-i]
         else:
              sinOut = sino
         return sinOut
    
    def crop_img(self,img,crop_factor=None):
         if crop_factor is None:
              crop_factor = self.sinogram.radialBinCropfactor
         if crop_factor!=0:
              i =  int(np.ceil(img.shape[0]*crop_factor/2.0)*2)//2    
              imgOut = img[i:img.shape[0]-i, i:img.shape[1]-i]
         else:
              imgOut = img
         return imgOut
    def uncrop_img(self,img):
         W = self.sinogram.nRadialBins_orig
         i = (W - img.shape[1])//2 
         imgOut = np.zeros((W,W,img.shape[2]),dtype=img.dtype)
         imgOut[i:W-i, i:W-i,:] = img 
         return imgOut         
    
    def read_sino(self,flname,num_planes=None,dtype='float32'):
         if num_planes is None:
              num_planes = self.sinogram.totalNumberOfSinogramPlanes
         sino_size = [self.sinogram.nRadialBins_orig, self.sinogram.nAngularBins, num_planes]   
         return np.fromfile(flname,dtype,np.prod(sino_size)).reshape(sino_size, order='F')

    def get_e7sino(self,path):
         prompts = self.crop_sino(self.read_sino(path+self.engine.bar+'emis_00.s'))
         randoms = self.crop_sino(self.read_sino(path+self.engine.bar+'smoothed_rand_00.s'))
         ncf = self.crop_sino(self.read_sino(path+self.engine.bar+'norm3d_00.a'))*self.get_gaps()
         acf = self.crop_sino(self.read_sino(path+self.engine.bar+'acf_00.a'))
         scatters = self.iSSRB(self.crop_sino(self.read_sino(path+self.engine.bar+'scatter_estim2d_000000.s',2*self.scanner.nCrystalRings-1)))

         NF = 1/ncf
         NF[np.isinf(NF)]=0
         AN = (1/acf)*NF
         RS = ncf*acf*(randoms + NF*scatters)
         return prompts, AN, RS          
         
         
         
         
         
         
         
         
         