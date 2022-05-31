#%%
import numpy


import matplotlib.pyplot as pl
import numpy
import h5py
import cv2
from Red_Neuronal import *

r=RedNeuronal()
r.cargar_parametros('Reconocimiento de Digitos\Files\\theta_digitos.h5')
#test=cv2.imread('../Files/test/testgrande.jpg')

#test=test[:,:,:]
def ProcessImg(Img):
        GrayImage=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
        GaussianFilter=cv2.GaussianBlur(GrayImage,(5,5),0)

        ret,imagen_bn=cv2.threshold(GaussianFilter,90,255,cv2.THRESH_BINARY_INV)
        grupos,_=cv2.findContours(imagen_bn.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        ventanas= [cv2.boundingRect(g) for g in grupos]
        a=[]
        for v in ventanas:
                #cv2.rectangle(imagen_bn,(v[0],v[1]),(v[0]+v[2],+v[1]+v[3]),(255,0,0),2)
                espacio=int(v[3]*1.6)
                p1=int((v[1]+v[3]//2))-espacio//2
                p2=int((v[0]+v[2]//2))-espacio//2

                digito=imagen_bn[p1:p1+espacio,p2:p2+espacio]
                if p2>0 and p1>0 and espacio>40:
                        height,width=digito.shape
                        rescaled = cv2.resize(digito, (0, 0), fx=20/width, fy=20/height)
                        rescaled=rescaled.T
                        rescaled=rescaled.reshape(1,-1)
                        identified=r.predecir(rescaled)
                        #print(identified)
                        cv2.putText(Img,str(identified),(v[0],v[1]-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                pl.imshow(Img)
        return Img


camara=cv2.VideoCapture(0)
while True:
    success,img=camara.read()
    img=ProcessImg(img)
    cv2.imshow('Video',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()
#a [2,5,0,1,2,5,9,8]

