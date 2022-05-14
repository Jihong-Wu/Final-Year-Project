import numpy as np
import matplotlib.pyplot as plt
from dmrg_infinite import DMRG,solve_E0,Sz_value, exact_infinite_E0, entanglement_entropy

if __name__ == "__main__":
    # TFIM
    L = 2;
    J = 1.;
    hs=np.linspace(0.01,2,20)
    Nh= len(hs)
    chi = 100;
    sweeps = 50;
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.eye(2)

    E0s=[]
    Szs=[]
    Entropys=[]
    E0s_exact=[]
    fid=[]

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
        #print("DMRG_infinete ", "L=", L, " h=", h)
        sigma_z = np.mean(Sz_value(L, Bs, Ss))
        #print("<sifma_z>=", sigma_z)
        E0 = np.mean(solve_E0(L, Bs, Ss, J, h))
        #print("E0=", E0)
        E_exact = exact_infinite_E0(J, h)
        #print("E_exact=", E_exact)
        entropy=np.mean(entanglement_entropy(Ss,L))
        #print("entropy=",entropy)
        Bc=Bs[0].conj()
        inner=np.tensordot(Bs[0],Bc,axes=[[0,1,2],[0,1,2]])

        E0s.append(E0)
        Szs.append(sigma_z)
        Entropys.append(entropy)
        E0s_exact.append(E_exact)
        fid.append(inner)


    x = [hs[i] / J for i in range(Nh)]
    plt.plot(x, Entropys, color='lightcoral', linewidth=3.0, linestyle='-', label='entanglement')
    plt.plot(x, fid, '+-', color='burlywood', markersize=10, label='fidelity')
    plt.show()