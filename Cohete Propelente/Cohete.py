#%%
from abc import ABC
from EulerHeuz import heuz,euler,RungeKuta
import numpy
import matplotlib.pyplot as pl

def Mp(t,mp0,dmpdt):
    ans=mp0-dmpdt*t
    if ans<0:
        ans=0
    return ans




def dep_pres(t, ci,mc,g,Ve,k,r,P,dmpdt):  #ci[v0,y0]
    masaProp=ci[2]#Mp(t,mp0,dmpdt)
    if(ci[2]<=0):
        dmpdt=0
    a=-g
    b=(abs(dmpdt)*Ve)/(mc+masaProp)
    c=-(k*P*numpy.pi*(r**2)*abs(ci[0])*ci[0])/(2*(mc+masaProp))
    dvdt = a+b+c
    dydt=ci[0]
    return numpy.array([dvdt,dydt,dmpdt])

mp0=100
dmpdt=-20
ci=numpy.array([0,0,mp0])
r=0.5
k=0.15
Ve=325
P=1.091
mc=50
g=9.81

Y,tiempo=heuz(0, 40, ci, 0.1, dep_pres, mc,g,Ve,k,r,P,dmpdt)

pl.plot(tiempo,Y[:,0],label="Velocidad m/s")
pl.plot(tiempo,Y[:,1],label="Altura",color="red")
pl.plot(tiempo,Y[:,2],label="Masa propelente",color="green")
pl.legend()
pl.grid()
print(max(Y[:,1]))
print(max(Y[:,0]))