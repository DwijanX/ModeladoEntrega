import numpy



def RungeKuta(t0, tf, ci, dt, direccion, *args):
    futuros = []
    tiempos = []
    while True:
        futuros.append(ci)
        tiempos.append(t0)
        if (t0 + dt) > tf:
            dt = tf - t0
        k1 = direccion(t0,ci, *args)
        k2 = direccion((t0+dt), ci+ (dt*k1), *args)
        k3 = direccion((t0+2*dt), ci+ (dt*k2), *args)
        k4 = direccion((t0+3*dt), ci+ (dt*k3), *args)
        pendiente = 1/6 *(k1+2*k2+2*k3+k4)
        ci = ci + dt * pendiente 
        t0 = t0 + dt
        if (t0 >= tf):
            break
    return numpy.array(futuros), numpy.array(tiempos)

def heuz(t0, tf, ci, dt, direccion, *args):
    futuros = []
    tiempos = []
    while True:
        futuros.append(ci)
        tiempos.append(t0)
        if (t0 + dt) > tf:
            dt = tf - t0
        k1 = direccion(t0,ci, *args)
        k2 = direccion((t0+dt), ci+ (dt*k1), *args)
        pendiente = 1/2 *(k1+k2)
        ci = ci + dt * pendiente 
        t0 = t0 + dt
        if (t0 >= tf):
            break
    return numpy.array(futuros), numpy.array(tiempos)

#
def euler(t0, tf, ci, dt, direccion, *args):
    futuros = []
    tiempos = []
    while True:
        futuros.append(ci)
        tiempos.append(t0)
        if (t0 + dt) > tf:
            dt = tf - t0
        pendiente = direccion(t0, ci, *args)
        ci = ci + dt * pendiente 
        t0 = t0 + dt
        if (t0 >= tf):
            break
    return numpy.array(futuros), numpy.array(tiempos)



def Convection(u0,nt,dt,dx):
    u=u0.copy()
    history=[u0.copy()]
    for n in range(1,nt):
        u[1:]=u[1:]-u[1:]*(dt/dx)*(u[1:]-u[:-1])
        history.append(u.copy())
    return history