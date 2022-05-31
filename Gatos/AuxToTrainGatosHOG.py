#%%
import h5py
import numpy
from Red_Neuronal  import *
import matplotlib.pyplot as pl
import cv2
from skimage.feature import hog

data=h5py.File("Python\Modelado\Files\gatillos_train.h5",'r')
#data=h5py.File("../Files/gatillos_train.h5",'r')
X=data["train_set_x"][:]
y=data["train_set_y"][:]
n=198
#pl.imshow(X[n][:])
XProcessed=numpy.array([])
XProcessed=numpy.array([])
def processImgHOG(Img):
    fd, hog_image = hog(Img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    #fd, hog_image = hog(Img, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return hog_image.reshape(1,-1)

XProcessed=numpy.vstack(numpy.array([processImgHOG(x) for x in X]))



epoca=1

r=RedNeuronal()
r.lambda_=10
r.capa1=64*64
r.capa2=128
r.capa3=2
#r.inicializar()
r.cargar_parametros("Python\Modelado\Files\HOG\ParamsGatos_"+str(epoca)+".h5")
#r.cargar_parametros("Python\Modelado\Files\HOG\ParamsGatos_Best.h5")

r.fit(XProcessed,y)
r.setProcessImg(processImgHOG)
for i in range(1,15):
    print("\nEpoca=",i+1)
    r.entrenar()
    Suc,Err=r.ProbarLambda2("Python\Modelado\Files\gatillos_test.h5","test_set_x","test_set_y")
    print("suc",Suc)
    print("Err",Err)
    r.GuardarParams("Python\Modelado\Files\HOG\ParamsGatos_"+str(i+1)+".h5")

#image PIL SKIMAGE