#%%
import cv2
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as pl

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
model_ft = torch.load("Gatos\Files\mi_modeloDeGatos.pt")


def ProbarLambda(FileDirection,XName,YNAME):
        ConfusionMatrix=numpy.array([[0,0],[0,0]])
        ErrorCount=0
        SuccessCount=0
        DataFile=h5py.File(FileDirection,'r')
        X=DataFile[XName][:]
        Y=DataFile[YNAME][:]
        counter=0
        for x in X:
            digito = transform(x)
            digito.unsqueeze_(dim=0)
            digito = Variable(digito)
            digito = digito.view(digito.shape[0], -1)
            ans = F.log_softmax(model_ft(digito))
            if(ans.argmax().item()!=Y[counter] ):
                ConfusionMatrix[Y[counter]][0]+=1
                ErrorCount+=1
            else:
                ConfusionMatrix[Y[counter]][1]+=1
                SuccessCount+=1
            counter+=1
            
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
    



Suc,Err=ProbarLambda("Gatos\Files\gatillos_test.h5","test_set_x","test_set_y")

print("aciertos",Suc)
print("fallos",Err)





















#%%

