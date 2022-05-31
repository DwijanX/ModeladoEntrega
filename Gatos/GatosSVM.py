#%%
import numpy
from sklearn.svm import SVC
import h5py
import matplotlib.pyplot as pl
import cv2
from skimage.feature import hog


def ProbarLambda(FileDirection,XName,YNAME):
        ConfusionMatrix=numpy.array([[0,0],[0,0]])
        ErrorCount=0
        SuccessCount=0
        DataFile=h5py.File(FileDirection,'r')
        X=DataFile[XName][:]
        Y=DataFile[YNAME][:]
        for x in X:
            ans=clasificador.predict(processImgHOG(x))
            if(ans!=Y[(ErrorCount+SuccessCount)] ):
                ConfusionMatrix[Y[(ErrorCount+SuccessCount)]][0]+=1
                ErrorCount+=1
            else:
                ConfusionMatrix[Y[(ErrorCount+SuccessCount)]][1]+=1
                SuccessCount+=1
        Precision=ConfusionMatrix[1][1]/(ConfusionMatrix[1][1]+ConfusionMatrix[1][0])
        Recall=ConfusionMatrix[1][1]/(ConfusionMatrix[1][1]+ConfusionMatrix[0][0])
        F1=2*((Precision*Recall)/(Precision+Recall))
        Accuracy=(ConfusionMatrix[1][1]+ConfusionMatrix[0][1])/(ConfusionMatrix[1][1]+ConfusionMatrix[1][0]+ConfusionMatrix[0][1]+ConfusionMatrix[0][0])
        print(ConfusionMatrix)
        print("Precision",Precision*100)
        print("Recall",Recall*100)
        print("F1",F1*100)
        print("Accuracy",Accuracy*100)
        
        return SuccessCount/(ErrorCount+SuccessCount),ErrorCount/(ErrorCount+SuccessCount)
    

#data=h5py.File("Python\Modelado\Files\gatillos_train.h5",'r')
data=h5py.File("Gatos\Files\gatillos_train.h5",'r')
X=data["train_set_x"][:]
y=data["train_set_y"][:]

XProcessed=numpy.array([])
def processImgHOG(Img):
    fd, hog_image = hog(Img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    #fd, hog_image = hog(Img, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return hog_image.reshape(1,-1)

XProcessed=numpy.vstack(numpy.array([processImgHOG(x) for x in X]))

clasificador = SVC(kernel="poly",C=1)
clasificador.fit(XProcessed,y)

#Suc,Err=ProbarLambda("Python\Modelado\Files\gatillos_test.h5","test_set_x","test_set_y")
Suc,Err=ProbarLambda("Gatos\Files\gatillos_test.h5","test_set_x","test_set_y")

print("aciertos",Suc)
print("fallos",Err)

#%%