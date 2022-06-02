#%%
import cv2
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as pl

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
model_ft = torch.load("Reconocimiento de Digitos\Files\mi_modelo_digitos.pt")
#model_ft = torch.load("Python\Modelado\Files\mi_modelo_digitos.pt")


#imagen = cv2.imread("Python\Modelado\Files\test\testgrande.jpg")
imagen = cv2.imread("Reconocimiento de Digitos\Files\\testgrande.jpg")
imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagenGris = cv2.GaussianBlur(imagenGris, (5, 5), 0)
ret, imagenBN = cv2.threshold(imagenGris, 90, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Digitos", imagenBN)
cv2.waitKey()
grupos, _ = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ventanas = [cv2.boundingRect(g) for g in grupos]

for g in ventanas:
    cv2.rectangle(imagen, (g[0], g[1]), (g[0] + g[2], g[1] + g[3]), (255, 0, 0), 2)
    l = int(g[3] * 1.6)
    p1 = int(g[1] + g[3] // 2) - l // 2
    p2 = int(g[0] + g[2] // 2) - l // 2

    digito = imagenBN[p1: p1+l, p2: p2+l]
    pl.imshow(digito)
    
    digito = cv2.resize(digito, (28, 28), interpolation=cv2.INTER_AREA)
    digito = cv2.dilate(digito, (3, 3,))

    digito = transform(digito)
    digito.unsqueeze_(dim=0)
    digito = Variable(digito)

    digito = digito.view(digito.shape[0], -1)
    predict = F.softmax(model_ft(digito), dim=1)

    cv2.putText(imagen, str(predict.argmax().item()), (g[0], g[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))


a=cv2.imshow("Digitos", imagen)
cv2.waitKey()
cv2.destroyAllWindows()