import numpy as np
import matplotlib.pyplot as plt
import math
from dmrg_infinite import DMRG,solve_E0,Sz_value, exact_infinite_E0, entanglement_entropy
from dmrg_finite import DMRG as DMRGf
from dmrg_finite import Sz_value as Sz_valuef

if __name__ == "__main__":
    J=1.
    hs=np.linspace(0.01,2,50)
    Nh=len(hs)
    Szs=np.zeros(Nh)
    a=0
    #exact Sz0
    q=np.linspace(0,np.pi,320)
    sq=np.sin(q)
    Nq=len(q)
    theta_q0=np.zeros(Nq)
    for h in hs:
        epsi_0=h-np.cos(q)
        for i in range(Nq):
            if epsi_0[i]>0:
                theta_q0[i]=math.atan(abs(sq[i]/epsi_0[i]))
            else:
                theta_q0[i]=np.pi-math.atan(abs(sq[i]/epsi_0[i]))
        sigz=0
        for i in range(Nq):
            sigz=sigz+np.cos(theta_q0[i])
        sigz=sigz/Nq
        Szs[a]=sigz
        a=a+1


    #DMRG
    L=2
    chi = 100
    sweeps = 50
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.eye(2)

    E0s = []
    Szsi = []
    Entropys = []
    E0s_exact = []

    for h in hs:
        print("h=",h)
        # initialize MPS
        B = np.zeros([1, 2, 1], dtype=float)
        B[0, 0, 0] = 1.
        S = np.ones([1], dtype=float)
        Bs = [B.copy() for i in range(L)]
        Ss = [S.copy() for i in range(L)]

        # initialize MPO
        mpo = []
        for i in range(L):
            w = np.zeros((3, 3, 2, 2), dtype=float)
            w[0, 0] = w[2, 2] = id
            w[0, 1] = sx
            w[0, 2] = -h * sz
            w[1, 2] = -J * sx
            mpo.append(w)

        Bs, Ss = DMRG(L, Bs, Ss, mpo, chi, sweeps)
        # calculations
        sigma_z = np.mean(Sz_value(L, Bs, Ss))
        #E0 = np.mean(solve_E0(L, Bs, Ss, J, h))
        #E_exact = exact_infinite_E0(J, h)
        #entropy = np.mean(entanglement_entropy(Ss, L))
        #E0s.append(E0)
        Szsi.append(sigma_z)
        #Entropys.append(entropy)

    print(Szsi)
    np.savetxt('sz.txt', Szsi)
    np.savetxt('hs.txt',hs)
    dsz=np.zeros(Nh-1)
    dh=hs[1]-hs[0]
    xx=np.zeros(Nh-1)
    for i in range(Nh-1):
        dsz[i]=(Szsi[i+1]-Szsi[i])/dh
        xx[i]=hs[i]

    plt.plot(xx, dsz, 'ro-')
    plt.xlabel('h/J', fontsize=15)
    plt.ylabel('X_z', fontsize=15)
    plt.xlim(0, 2)
    plt.show()

    # DMRG finite
    '''
    J = 1.;
    chi = 30;
    sweeps = 20;
    Ls = [20, 30, 50]
    Szf1 = [];
    Szf2 = [];
    Szf3 = []
    a = 0;
    for L in Ls:
        print("runing")
        Eg = []
        for h in hs:
            # initialize MPS
            B = np.zeros([1, 2, 1], dtype=float)
            B[0, 0, 0] = 1.
            S = np.ones([1], dtype=float)
            Bs = [B.copy() for i in range(L)]
            Ss = [S.copy() for i in range(L)]

            # initialize MPO
            mpo = []
            for i in range(L):
                w = np.zeros((3, 3, 2, 2), dtype=float)
                w[0, 0] = w[2, 2] = id
                w[0, 1] = sx
                w[0, 2] = -h * sz
                w[1, 2] = -J * sx
                mpo.append(w)

            Bs, Ss = DMRGf(L, Bs, Ss, mpo, chi, sweeps)
            E0 = np.mean(Sz_valuef(L, Bs, Ss))
            Eg.append(E0)
        a = a + 1
        if a == 1: Szf1 = Eg
        if a == 2: Szf2 = Eg
        if a == 3: Szf3 = Eg


'''


    #plt.plot(hs, Szs, color='lightcoral', linewidth=3.0, linestyle='-', label='Exact')
    #plt.plot(hs,Szsi,'+',color='mediumturquoise',markersize=10,label='DMRG-infinite')
    #plt.plot(hs, Szf1, '+-', color='burlywood', markersize=10, label='DMRG-finite20')
    #plt.plot(hs, Szf2, '+-', color='mediumpurple', markersize=10, label='DMRG-finite30')
    #plt.plot(hs, Szf3, '+-', color='lightgreen', markersize=10, label='DMRG-finite40')
    #plt.legend(loc='best')
    #plt.xlabel('h/J', fontsize=15)
    #plt.ylabel('sigma_z', fontsize=15)
    #plt.xlim(0, 2)
    #plt.show()



