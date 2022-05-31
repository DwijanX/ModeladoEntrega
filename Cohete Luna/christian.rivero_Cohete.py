from vpython import *
#GlowScript 3.2 VPython

#display(width=600, height=600, center=vector(0, 12, 0), background=color.black)

#masas
#tierra = sphere(pos=vector(0,0,0),radius=6.370, texture=textures.earth)
#luna= sphere(pos=vector(20,30,10),radius=1.7374, color=color.white)
#cohete = sphere(pos=vector(5,10,0),radius=1.000, color=color.green)

G=6.667*((10)**(-11))# N M^2/kg^2
DTierraLuna= 384400 #m
DTierraCohete= 104400 #m
rTierra=6370 #m
rLuna=1737.4 #m
mTierra=5.972*(10**24) #kg
mLuna=7.349*(10**22) #kg
periodo_Lunar=29#*24*3600 horas

tierra = sphere(pos = vec(0,0,0),radius = 10*rTierra, texture=textures.earth)
luna = sphere(pos = vec(0,DTierraLuna,0),make_trail =True, radius = 10*rLuna, color=color.white)
cohete=sphere(pos=vector(0,DTierraCohete,0),make_trail =True,radius=8*rLuna,color=color.red)

def get_movLunar(t,t0):
    theta=t0+2*pi*t/periodo_Lunar
    posx=DTierraLuna*cos(theta)
    posy=DTierraLuna*sin(theta)
    return vec(posx,posy,0)

    
def getAceleracion(t,objpos,mt,ml):
    VectorLuna=get_movLunar(t,0)-objpos
    VectorTierra=-objpos
    dvdt=G*VectorTierra*mt/(VectorTierra.mag)**3+G*ml*VectorLuna/(VectorLuna.mag)**3
    return dvdt
    
def GetVeloInicial(MasaObj,ArrayDistance):
    a=sqrt(G*MasaObj/(ArrayDistance.mag))
    return a
t = 0

MagnitudVelocidadNave = GetVeloInicial(mTierra,(tierra.pos-cohete.pos)) 
velocidad=vec(MagnitudVelocidadNave+13000,0,0)
print(MagnitudVelocidadNave)
dt=1
print(dt)
Resultados=[]
t=0
while(t<10*periodo_Lunar):
    rate(30)
    k1=getAceleracion(t,cohete.pos,mTierra,mLuna)
    aux1=velocidad + k1*dt
    aux2=cohete.pos + aux1*dt
    k2=getAceleracion(t,aux2,mTierra,mLuna)
    pendiente=(k1+k2)/2
    velocidad = velocidad + k1*dt
    luna.pos=get_movLunar(t,0)
    cohete.pos = cohete.pos + velocidad*dt
    t = t + dt
    