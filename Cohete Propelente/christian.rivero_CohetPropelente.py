from vpython import *
#GlowScript 3.2 VPython

#display(width=600, height=600, center=vector(0, 12, 0), background=color.black)



cohete=sphere(pos=vector(0,0,0),make_trail =False,radius=100,color=color.red)
Base=box(pos=vector(0,-30,0),size=vector(1000,100,1000),color=color.blue)




mp0=100
dmpdt=-20
r=0.5
k=0.15
Ve=325
P=1.091
mc=50
g=9.81
masaProp=mp0

velocidad=0
dt=0.1
Resultados=[]
t=0
while(t<40):
    rate(30)
    if(masaProp<=0):
        dmpdt=0
    a=-g
    b=(abs(dmpdt)*Ve)/(mc+masaProp)
    c=-(k*P*pi*(r**2)*abs(velocidad)*velocidad)/(2*(mc+masaProp))
    k1dvdt=a+b+c
    k1dydt=velocidad
    
    cohete.pos=cohete.pos+vec(0,k1dydt*dt,0)
    velocidad=velocidad+k1dvdt*dt
    masaProp=masaProp+dt*dmpdt
    
    t = t + dt
    