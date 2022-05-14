import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    delta=0.5
    l=8
    v=np.pi/2*np.sqrt(1-delta**2)/np.arccos(delta)
    K=np.pi/2/(np.pi-np.arccos(delta))
    t=np.linspace(0.01,10,1000)
    Nt=len(t)
    Szcor=np.zeros(Nt)
    for i in range(Nt):
        aa=l*l-(2*v*t[i])**2
        bb=(2*v*t[i])**2*l*l
        cc=(1/K/K-1)/8
        sz=1/np.sqrt(l)*((abs(aa/bb))**cc)
        Szcor[i]=sz
        #print("t=",t[i],"    aa=",aa,"    bb=",bb,"    cc=",cc,"    sz=",sz)

    #print(Szcor)
    print(Szcor[Nt-1])
    plt.plot(t,Szcor)
    plt.show()

