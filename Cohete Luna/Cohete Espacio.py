#%%
from cmath import sqrt

from EulerHeuz import heuz,euler,RungeKuta
import numpy
import matplotlib.pyplot as pl

G=6.667*(10)**(-11)# N*M^2/kg^2
MasaTierra=5.972*(10**24) #kg
MasaLuna=7.349*(10**22) #kg
DTierraCohete= 104400
dist_Luna=384400  #dist_Luna=1.5e7
periodo_Lunar=29#*24*3600



def get_movLunar(t,t0):
    theta=t0+2*numpy.pi*t/periodo_Lunar
    posx=dist_Luna*numpy.cos(theta)
    posy=dist_Luna*numpy.sin(theta)
    return numpy.array([posx,posy])




#ci=>[Posicion,Velocidad]
def getDirCohete(t,ci,mt,ml):
    dvdt=getAceleracion(t,ci,mt,ml)
    dxdt=ci[1]
    return numpy.array([dxdt,dvdt])


def getAceleracion(t,ci,mt,ml):
    auxPosLuna=get_movLunar(t,0)
    VectorLuna=auxPosLuna-ci[0]
    VectorTierra=-ci[0]
    dvdt=G*(VectorTierra*mt/numpy.linalg.norm(VectorTierra)**3+ml*VectorLuna/numpy.linalg.norm(VectorLuna)**3)
    return dvdt

def GetVeloInicial(MasaObj,ArrayDistance):
    a=sqrt(G*MasaObj/numpy.linalg.norm(ArrayDistance))
    return a

PosLuna=[]

posTierra=numpy.array([0,0])
posLuna=numpy.array([0,dist_Luna])
posCohete=numpy.array([0,DTierraCohete])
MagnitudVelocidadNave = GetVeloInicial(MasaTierra,(posTierra-posCohete)) 
ci=numpy.array([numpy.array([0,DTierraCohete]),numpy.array([MagnitudVelocidadNave+13000,0])])
dt=0.1



Y,tiempo=heuz(0,100*periodo_Lunar,ci,dt,getDirCohete,MasaTierra,MasaLuna)

PosicionesLuna=get_movLunar(tiempo,0)

posLuna=numpy.array(posLuna)
pl.plot(Y[:,0][:,0],Y[:,0][:,1],label="cohete",color="red")
pl.plot(PosicionesLuna[0,:],PosicionesLuna[1,:],label="luna",color="blue")
pl.legend()
pl.grid()