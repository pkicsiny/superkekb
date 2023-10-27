import sys
import os
import numpy as np

class PyBeamBeam6D:
    _sinWeakXing = 0.0
    _cosWeakXing = 1.0
    _tanWeakXing = 0.0

    _nSlice = 1
    _factor = 0.0
    _table = None

    def __init__(self,nSlice=1,weakXing=0.0,xiY=0.0,
            #weakSigmaZ=0.8,weakSigmaE=1E-4,weakEmitX=3E-6,weakBetaX=0.4,weakAlphaX=0.0,weakEmitY=3E-6,weakBetaY=0.4,weakAlphaY=0.0,weakEtaX=0.0,weakEtaXP=0.0,weakEtaY=0.0,weakEtaYP=0.0,
            strongXing=None,strongSigmaZ=1.0,strongSigmaDPP=1.0,strongEmitX=1.0,strongBetaX=1.0,strongAlphaX=0.0,strongEmitY=1.0,strongBetaY=1.0,strongAlphaY=0.0,strongEtaX=0.0,strongEtaXP=0.0,strongEtaY=0.0,strongEtaYP=0.0):
        if nSlice%2 == 0:
            print('Number of slices must be off - ABORT\n') #TODO why ?
            exit()
        self.setWeakXing(weakXing)
        self._nSlice = nSlice
        self._sliceX = np.zeros(nSlice,dtype=float)
        self._sliceY = np.zeros(nSlice,dtype=float)
        self._sliceZ = np.zeros(nSlice,dtype=float)
        self._sliceSigmaX2 = np.zeros(nSlice,dtype=float)
        self._sliceSigmaXP2 = np.zeros(nSlice,dtype=float)
        self._sliceSigmaY2 = np.zeros(nSlice,dtype=float)
        self._sliceSigmaYP2 = np.zeros(nSlice,dtype=float)

        SIGXXN=np.sqrt(strongEmitX*strongBetaX)
        SIGYYN=np.sqrt(strongEmitY*strongBetaY)
        self._factor=xiY#/strongBetaY*2.0*np.pi*SIGYYN*(SIGYYN+SIGXXN)/self._nSlice
        print(self._factor)
        
        ####quick fix#####
        if SIGXXN == SIGYYN:
            strongBetaY *= 1.001
        ##################

        if strongXing == None:
            strongXing = weakXing

        PySBC.stsld(self._sliceX,self._sliceY,self._sliceZ,strongXing,strongSigmaZ,strongSigmaDPP,strongEmitX,strongBetaX,strongAlphaX,strongEmitY,strongBetaY,strongAlphaY,strongEtaX,strongEtaXP,strongEtaY,strongEtaYP,self._sliceSigmaX2,self._sliceSigmaXP2,self._sliceSigmaY2,self._sliceSigmaYP2,nSlice)
        self._table = np.zeros((4,40,31),dtype=complex,order='F')
        PySBC.seterf(self._table)

    def setWeakXing(self,weakXing):
        self._sinWeakXing = np.sin(weakXing)
        self._cosWeakXing = np.cos(weakXing)
        self._tanWeakXing = np.tan(weakXing)

    def boost(self,x,xp,y,yp,z,dpp):
        PySBC.boost(x,y,z,xp,yp,dpp,self._sinWeakXing,self._cosWeakXing,self._tanWeakXing,len(x))

    def boostInverse(self,x,xp,y,yp,z,dpp):
        PySBC.boosti(x,y,z,xp,yp,dpp,self._sinWeakXing,self._cosWeakXing)

    def kick(self,x,xp,y,yp,z,dpp):
        #dpp0 = np.copy(dpp)
        PySBC.sbc(x,y,z,xp,yp,dpp,self._sliceX,self._sliceY,self._sliceZ,self._sliceSigmaX2,self._sliceSigmaXP2,self._sliceSigmaY2,self._sliceSigmaYP2,self._factor,self._table,np.shape(x)[0],self._nSlice)
        #print('SBC kick',(dpp-dpp0))

    def plotStrongBeam(self):
        print(self._sliceX,self._sliceY,self._sliceZ)
        print(self._sliceSigmaX2,self._sliceSigmaXP2,self._sliceSigmaY2,self._sliceSigmaYP2)

        plt.figure(0)
        plt.plot(self._sliceZ,self._sliceX,'-x')
        plt.plot(self._sliceZ,self._sliceY,'-x')
        plt.figure(1)
        plt.plot(self._sliceZ,self._sliceSigmaX2,'-x',color='b')
        plt.plot(self._sliceZ,self._sliceSigmaXP2,'--x',color='b')
        plt.plot(self._sliceZ,self._sliceSigmaY2,'-x',color='g')
        plt.plot(self._sliceZ,self._sliceSigmaYP2,'--x',color='g')
