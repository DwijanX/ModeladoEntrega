#%%
import numpy
from sklearn.svm import SVC
import h5py
import matplotlib.pyplot as pl
import cv2


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
                        rescaled=rescaled.reshape(1,-1)
                        identified=clasificador.predict(rescaled)
                        #pl.imshow(rescaled.reshape(20,20))
                        #print(identified)
                        cv2.putText(Img,str(identified),(v[0],v[1]-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                #pl.imshow(Img)
        return Img
#data = h5py.File("Python\Modelado\Files\digitos.h5", "r")

data = h5py.File("Reconocimiento de Digitos\Files\digitos.h5", "r")
test=cv2.imread('Reconocimiento de Digitos\Files\testgrande.jpg')

X = numpy.array(data["X"][:])
Y = numpy.array(data["y"][:])

print(X.shape)
testn=1500
X=X.reshape(5000,20,20)
for i in range(5000):
        X[i]=X[i].T
X=X.reshape(5000,400)
clasificador = SVC(kernel="poly", gamma=10,C=1000)
clasificador.fit(X,Y)


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    frame=ProcessImg(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
#%%