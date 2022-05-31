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
n=198
#pl.imshow(X[n][:])
XProcessed=numpy.array([])
XProcessed=numpy.array([])
def processImgRGB(Img):
    return Img.reshape(1,-1)

XProcessed=numpy.vstack(numpy.array([processImgRGB(x) for x in X]))
epoca=1

r=RedNeuronal()
r.lambda_=1
r.capa1=64*64*3
r.capa2=128
r.capa3=2
#r.inicializar()
r.cargar_parametros("Gatos\ParametrosGenerados\GatoRGB\ParamsGatos_"+str(epoca)+".h5")
r.fit(XProcessed,y)
r.setProcessImg(processImgRGB)
for i in range(1,10):
    print("\nEpoca=",i+1)
    r.entrenar()
    Suc,Err=r.ProbarLambda("Gatos\Files\gatillos_test.h5","test_set_x","test_set_y")
    print("suc",Suc)
    print("Err",Err)
    r.GuardarParams("Gatos\ParametrosGenerados\GatoRGB\ParamsGatos_"+str(i+1)+".h5")

#epoca 9 70% godin


#r.GuardarParams("Python\Modelado\Files\miguardada.h5")
#r.GuardarParams("Python\Modelado\Files\trainGatosColor.h5")

#digito=X[834,:].reshape(1,-1)
#pl.imshow(digito.reshape(20,20).T)
#print(r.predecir(digito))



#revisar LBP HISTOGRAMA DE BLOQUE 
#revisar HOG HAAR
#image PIL SKIMAGE