#%%
import h5py
import numpy
from Red_Neuronal  import *
import matplotlib.pyplot as pl
import cv2

data=h5py.File("Gatos\Files\gatillos_train.h5",'r')
#data=h5py.File("../Files/gatillos_train.h5",'r')
X=data["train_set_x"][:]
y=data["train_set_y"][:]

#la 3 muy god
epoca=3
n=198
#pl.imshow(X[n][:])
#print("Epoca=",epoca)
XProcessed=numpy.array([])
def processImg(Img):
    GrayImage=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    GaussianFilter=cv2.GaussianBlur(GrayImage,(5,5),0)
    return GaussianFilter.reshape(1,-1)


XProcessed=numpy.vstack(numpy.array([processImg(x) for x in X]))

#print(XProcessed.shape)

r=RedNeuronal()
r.lambda_=0
r.capa1=64*64
r.capa2=110
r.capa3=2

#r.inicializar()
r.cargar_parametros("Gatos\ParametrosGenerados\Blanco_Negro\ParamsGatos_"+str(epoca)+".h5")
r.fit(XProcessed,y)
r.setProcessImg(processImg)
for i in range(3,15):
    print("\nEpoca=",i+1)
    r.entrenar()
    #Suc,Err=r.ProbarLambda("../Files/gatillos_test.h5","test_set_x","test_set_y")
    Suc,Err=r.ProbarLambda("Gatos\Files\gatillos_test.h5","test_set_x","test_set_y")
    print("suc",Suc)
    print("Err",Err)
    r.GuardarParams("Gatos\ParametrosGenerados\Blanco_Negro\ParamsGatos_"+str(i+1)+".h5")

#4 vio bien
#10 reconocio 16 gatos
#11 mejor aun
#12 peor
#13 a peor

#revisar LBP HISTOGRAMA DE BLOQUE 
#revisar HOG HAAR