#%%
import cv2
import matplotlib.pyplot as pl
from scipy.io import loadmat
# importamos la libreria sklear, implementacion del algoritmo de Support Vector machine
from sklearn.svm import SVC
import cv2
import numpy
from PIL import Image
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

data=loadmat('Reconocimiento de Lentes\Files\\rostros.mat')
data.keys()
X=data['images']
y=data['target']
y=y[0]
yLentes=[10,11,12,13,14,15,16,17,18,19,30,31,32,37,38,50,51,52,53,54]
ySLentes=[0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29]

XImg1 = data["images"][ySLentes]
XImg2 = data["images"][yLentes]
XImag=numpy.concatenate((XImg1,XImg2),axis=0)
YImg1=numpy.zeros(20)
YImg2=numpy.ones(20) 
YImag=numpy.concatenate((YImg1,YImg2),axis=0)

def processImg(Img):
    return Img.reshape(1,-1)

XImagReshaped=numpy.vstack(numpy.array([processImg(x) for x in XImag]))
clasificador = SVC(kernel="rbf", gamma=10,C=1)
clasificador.fit(XImagReshaped,YImag)



video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    gris, rostro = agrupa_rostro(frame)
    #print(rostro)
    i = 0
    for face in rostro:
        (x, y, w, h) = face
        espacio=int(h*1.6)
        p1=int((y+h//2))-espacio//2
        p2=int((x+w//2))-espacio//2
        JustFace=gris[p1:p1+espacio,p2:p2+espacio]
        
        JustFace=numpy.resize(JustFace,(64,64))
        if w > 100:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            identified=clasificador.predict(JustFace.reshape(1,-1))
            cv2.putText(frame,str(identified),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(49) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
