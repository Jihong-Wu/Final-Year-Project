import numpy as np
import matplotlib.pyplot as plt
from dmrg_infinite import DMRG,solve_E0,Sz_value, exact_infinite_E0, entanglement_entropy
from dmrg_finite import DMRG as DMRGf
from dmrg_finite import solve_E0 as solve_E0f
import time



if __name__ == "__main__":
    time_start=time.time()
    # TFIM
    L = 2;
    J = 1.;
    hs=np.linspace(0.01,2,20)
    Nh= len(hs)
    chi = 30;
    sweeps = 20;
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.eye(2)

    E0s=[]
    Szs=[]
    Entropys=[]
    E0s_exact=[]

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

        Bs, Ss = DMRG(L, Bs, Ss, mpo, chi, sweeps)
        # calculations
        #print("DMRG_infinete ", "L=", L, " h=", h)
        sigma_z = np.mean(Sz_value(L, Bs, Ss))
        #print("<sifma_z>=", sigma_z)
        E0 = np.mean(solve_E0(L, Bs, Ss, J, h))
        #print("E0=", E0)
        E_exact = exact_infinite_E0(J, h)
        #print("E_exact=", E_exact)
        entropy=np.mean(entanglement_entropy(Ss,L))
        #print("entropy=",entropy)
        E0s.append(E0)
        Szs.append(sigma_z)
        Entropys.append(entropy)
        E0s_exact.append(E_exact)


    #DMRG finite
    J=1.;
    chi=30; sweeps=20;
    Ls=[20,30,50]
    E0f1=[]; E0f2=[]; E0f3=[]
    a=0;
    for L in Ls:
        print("runing")
        Eg=[]
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
            E0 = np.mean(solve_E0f(L, Bs, Ss, J, h))
            Eg.append(E0)
        a=a+1
        if a==1: E0f1=Eg
        if a==2: E0f2=Eg
        if a==3: E0f3=Eg




    x=[hs[i]/J for i in range(Nh)]
    #plt.figure()
    plt.plot(x, E0s_exact, color='lightcoral', linewidth=3.0, linestyle='-', label='Exact')
    plt.plot(x,E0s,'+',color='mediumturquoise',markersize=10,label='DMRG-infinite')
    plt.plot(x,E0f1,'+-',color='burlywood',markersize=10,label='DMRG-finite20')
    plt.plot(x, E0f2, '+-', color='mediumpurple', markersize=10, label='DMRG-finite30')
    plt.plot(x, E0f3, '+-', color='lightgreen', markersize=10, label='DMRG-finite40')

    plt.legend(loc='best')
    plt.xlabel('h/J',fontsize=15)
    plt.ylabel('E0',fontsize=15)
    plt.xlim(0,2)
    plt.show()
    time_end=time.time()
    print("运行时间: "+str(time_start-time_end)+"秒")


