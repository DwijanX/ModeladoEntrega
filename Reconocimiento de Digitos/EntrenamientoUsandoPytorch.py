#%%
import torch
from time import time
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset
from H5DData import *

# carga de datos
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#train_loader = torch.utils.data.DataLoader(H5DData("Python\Modelado\Files\digitos_train.h5", transform), batch_size=64, shuffle=True)
train_loader = torch.utils.data.DataLoader(H5DData("Reconocimiento de Digitos\Files\digitos_train.h5", transform), batch_size=64, shuffle=True)
#test_loader = torch.utils.data.DataLoader(H5DData("Python\Modelado\Files\digitos_test.h5", transform), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(H5DData("Reconocimiento de Digitos\Files\digitos_test.h5", transform), batch_size=64, shuffle=True)

# topologia de la red
capa_entrada = 784
capas_ocultas = [128, 64]
capa_salida = 10

# capas ocultas con la funcion de activacion RELU
#capa salida softmax
modelo = nn.Sequential(nn.Linear(capa_entrada, capas_ocultas[0]), nn.ReLU(),
                       nn.Linear(capas_ocultas[0], capas_ocultas[1]), nn.ReLU(),
                       nn.Linear(capas_ocultas[1], capa_salida), nn.LogSoftmax(dim=1))

# entropia cruzada
j = nn.CrossEntropyLoss()

# entrenamiento de la red
optimizador = optim.Adam(modelo.parameters(), lr=0.003)
tiempo = time()
epochs = 1
for e in range(epochs):
    costo = 0
    for imagen, etiqueta in train_loader:
        
        imagen = imagen.view(imagen.shape[0], -1)
        optimizador.zero_grad()
        h = modelo(imagen.float())
        error = j(h, etiqueta.long())
        error.backward()
        optimizador.step()
        costo += error.item()
    else:
        print("Epoch {} - Funcion costo: {}".format(e, costo / len(train_loader)))
print("\nTiempo de entrenamiento (en minutes) =", (time() - tiempo) / 60)
#torch.save(modelo, 'Python\Modelado\Files\mi_modelo_digitos.pt')
torch.save(modelo, 'Reconocimiento de Digitos\Files\mi_modelo_digitos.pt')
