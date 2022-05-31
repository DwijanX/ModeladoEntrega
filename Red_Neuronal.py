from errno import errorcode
import h5py
import numpy
import scipy.optimize
import cv2
import matplotlib.pyplot as pl

class RedNeuronal:

    def __init__(self):
        self.X = None
        self.y = None
        self.theta1 = None
        self.theta2 = None
        self.lambda_ = 0
        self.capa1 = None
        self.capa2 = None
        self.capa3 = None
        self.ProcessImg=None

    def fit(self, x, y):
        self.X = x
        self.y = y

    def sigmoide(self, z):
        s = 1 / (1 + numpy.exp(-z))
        return s
    def inicializar(self,epsilon=0.12):
        self.theta1=numpy.random.rand(self.capa2,(self.capa1+1))*2*epsilon-epsilon
        self.theta2=numpy.random.rand(self.capa3,(self.capa2+1))*2*epsilon-epsilon
    def cargar_parametros(self, archivo):
        param = h5py.File(archivo, 'r')
        self.theta1 = param['Theta1'][:]
        self.theta2 = param['Theta2'][:]
        #print(self.theta1.shape)
        #print(self.theta2.shape)
    def GuardarParams(self,data):
        t=h5py.File(data,'w')
        t.create_dataset('Theta1',data=self.theta1)
        t.create_dataset('Theta2',data=self.theta2)
        t.close()

    def predecir(self, imagen):
        a1 = numpy.concatenate([numpy.ones((1, 1)), imagen], axis=1)
        a2 = self.sigmoide(a1.dot(self.theta1.T))
        a2 = numpy.concatenate([numpy.ones((a2.shape[0], 1)), a2], axis=1)
        a3 = self.sigmoide(a2.dot(self.theta2.T))
        return a3.argmax()
    def derivadaSigmoide(self,a):
        return self.sigmoide(a)*(1-self.sigmoide(a))
    def getCrossedEntropy(self,t):
        # re armado de los parametros segun topologia de la red
        t1 = numpy.reshape(t[0:self.capa2 * (self.capa1 + 1)], (self.capa2, (self.capa1 + 1)))
        t2 = numpy.reshape(t[self.capa2 * (self.capa1 + 1):], (self.capa3, (self.capa2 + 1)))

        # dimension de X
        m, n = self.X.shape

        # Computar h: front propagation
        a1 = numpy.concatenate([numpy.ones((m, 1)), self.X], axis=1)
        a2 = self.sigmoide(a1.dot(t1.T))
        a2 = numpy.concatenate([numpy.ones((a2.shape[0], 1)), a2], axis=1)
        h = self.sigmoide(a2.dot(t2.T))
        
        # vectorizar y
        y_vec = numpy.eye(self.capa3)[self.y.reshape(-1)]

        # computo del parametro de regularizacion, para contrarestar el sobre ajuste
        param_reg = (self.lambda_ / (2 * m)) * (numpy.sum(numpy.square(t1[:, 1:])) +
                                                numpy.sum(numpy.square(t2[:, 1:])))

        # computo del costo: entropia cruzada
        j = - 1 / m * numpy.sum(numpy.log(h) * y_vec + numpy.log(1 - h) * (1 - y_vec)) + param_reg

        # coputar el gradiente: back propagation
        # error en la ultima capa
        delta3 = h - y_vec
        # error en la penultima capa
        delta2 = delta3.dot(t2)[:, 1:] * self.derivadaSigmoide(a1.dot(t1.T))

        # computo errores en las capas acumulado
        delta_acum_1 = delta2.T.dot(a1)
        delta_acum_2 = delta3.T.dot(a2)

        # computo del gradiente
        grad1 = 1 / m * delta_acum_1
        grad2 = 1 / m * delta_acum_2

        #  penalización con el parametro de regularizacion
        grad1[:, 1:] = grad1[:, 1:] + (self.lambda_ / m) * t1[:, 1:]
        grad2[:, 1:] = grad2[:, 1:] + (self.lambda_ / m) * t2[:, 1:]

        # concatenar gradientes
        grad = numpy.concatenate([grad1.flatten(), grad2.flatten()])
        return j, grad
    
    def entrenar(self):
        j_grad=lambda p:self.getCrossedEntropy(p)
        theta_Inicial=numpy.concatenate([self.theta1.flatten(),self.theta2.flatten()])
        opciones={'maxiter':500}
        
        res=scipy.optimize.minimize(j_grad,theta_Inicial,jac=True,method="TNC",options=opciones)
        thetaOptimos=res.x
        self.theta1=thetaOptimos[0:(self.capa1+1)*self.capa2].reshape(self.capa2,self.capa1+1)
        self.theta2=thetaOptimos[(self.capa1+1)*self.capa2:].reshape(self.capa3,self.capa2+1)
    def setProcessImg(self,Method):
        self.ProcessImg=Method
    """
    def ProcessImg(self,Img):
        GrayImage=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
        GaussianFilter=cv2.GaussianBlur(GrayImage,(5,5),0)
        ret,imagen_bn=cv2.threshold(GaussianFilter,90,255,cv2.THRESH_BINARY_INV)
        return imagen_bn.reshape(1,-1)
        """
    def ProbarLambda(self,FileDirection,XName,YNAME):
        ConfusionMatrix=numpy.array([[0,0],[0,0]])
        ErrorCount=0
        SuccessCount=0
        DataFile=h5py.File(FileDirection,'r')
        X=DataFile[XName][:]
        Y=DataFile[YNAME][:]
        for x in X:
            ans=self.predecir(self.ProcessImg(x))
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
    def ProbarLambda2(self,FileDirection,XName,YNAME):
        ConfusionMatrix=numpy.array([[0,0],[0,0]])
        ErrorCount=0
        SuccessCount=0
        DataFile=h5py.File(FileDirection,'r')
        X=DataFile[XName][:]
        Y=DataFile[YNAME][:]
        for x in X:
            ans=self.predecir(self.ProcessImg(x))
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
        if Precision<0.65:
            self.lambda_+=15
        print(ConfusionMatrix)
        print("Precision",Precision*100)
        print("Recall",Recall*100)
        print("F1",F1*100)
        print("Accuracy",Accuracy*100)
        
        return SuccessCount/(ErrorCount+SuccessCount),ErrorCount/(ErrorCount+SuccessCount)
                

        




