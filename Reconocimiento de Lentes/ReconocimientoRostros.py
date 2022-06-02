#%%
from tkinter.tix import IMAGE
import cv2
import matplotlib.pyplot as pl
from scipy.io import loadmat
# importamos la libreria sklear, implementacion del algoritmo de Support Vector machine
from sklearn.svm import SVC
import cv2
import numpy
from PIL import Image
from matplotlib import cm

def agrupa_rostro(frame):
    haar_xml =cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    modelo = cv2.CascadeClassifier(haar_xml)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = modelo.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(64, 64),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return gris, rostros

data=loadmat('./Files/rostros.mat')
data.keys()
X=data['images']
y=data['target']
y=y[0]
yLentes=[10,11,12,13,14,15,16,17,18,19,30,31,32,37,38,50,51,52,53,54,55,56,57,58,59,63,64,69,120,121,124,125,126,127,128,129,30,131,132,133,137,139,164,165,166,180,181,185,191,197]
ySLentes=[0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29,33,34,35,36,39,40,41,42,43,44,45,46,47,48,49,60,61,62,65,66,67,68,70,71,72,73,74,75,80,85,86,87,90,95,100,186,193]

XImg1 = data["images"][ySLentes]
XImg2 = data["images"][yLentes]
XImag=numpy.concatenate((XImg1,XImg2),axis=0)
YImg1=numpy.zeros(len(ySLentes))
YImg2=numpy.ones(len(yLentes)) 
YImag=numpy.concatenate((YImg1,YImg2),axis=0)

def processImg(Img):
    return Img.reshape(1,-1)

XImagReshaped=numpy.vstack(numpy.array([processImg(x) for x in XImag]))
clasificador = SVC(kernel="poly",gamma=2,C=1)
clasificador.fit(XImagReshaped,YImag)

def ProbarLambda(X,Y):
        ConfusionMatrix=numpy.array([[0,0],[0,0]])
        ErrorCount=int(0)
        SuccessCount=int(0)
        for x in X:
            ind=ErrorCount+SuccessCount
            ans=clasificador.predict(x.reshape(1,-1))
            if(ans!=Y[ind]):
                ConfusionMatrix[int(Y[ind])][0]+=1
                ErrorCount+=1
            else:
                ConfusionMatrix[int(Y[ind])][1]+=1
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
    
#print(YImag[0])
#Suc,Err=ProbarLambda(XImagReshaped,YImag)

#print("aciertos",Suc)
#print("fallos",Err)


"""
img=cv2.imread('.\\Files\\test.jpg')
gris, rostro = agrupa_rostro(img)
#print(rostro)
i = 0
for face in rostro:
    (x, y, w, h) = face

    if x+w>0 and y + h>0 :
        JustFace = gris[y:y + h,x:x+w]
        pilImg = Image.fromarray(JustFace)
        JustFace = pilImg.resize((64,64))
        JustFace = numpy.array(JustFace)
        pl.imshow(JustFace)
        pl.show()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        identified=clasificador.predict(JustFace.reshape(1,-1))
        print(identified)
        cv2.putText(img,str(identified),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
        pl.imshow(img)
        pl.show()
"""
#%%               

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    gris, rostro = agrupa_rostro(frame)
    i = 0
    for face in rostro:
        (x, y, w, h) = face
        if x+w>0 and y + h>0 :
            JustFace = gris[y:y + h,x:x+w]
            pilImg = Image.fromarray(JustFace)
            JustFace = pilImg.resize((64,64))
            JustFace = numpy.array(JustFace)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            identified=clasificador.predict(JustFace.reshape(1,-1))
            print(identified)
            cv2.putText(frame,str(identified),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
        
    cv2.imshow('Video', frame)

    if cv2.waitKey(49) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

#%%